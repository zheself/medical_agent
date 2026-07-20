"""
orchestrator.py — 主调度器

把所有模块串起来，对外暴露统一的 answer(query) 接口。

调用链:
    User Query
      ↓
    Router (复杂度判断、模型路由)
      ↓
    [Episodic Memory 异步检索 + Speculative Prefetch]
      ↓
    Planner (生成 DAG 计划)
      ↓
    Tool DAG 并行执行
      ↓
    Draft Answer 合成
      ↓
    分级 Verifier (L1 → L2 → L3 按需)
      ↓
    Final Answer + Memory 异步写入

实现状态: ✅ 完整调度逻辑
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple

from .agents.planner import PlannerAgent
from .memory.episodic_memory import EpisodicMemory
from .memory.semantic_memory import SemanticMemory
from .memory.working_memory import (
    WorkingMemory,
    WorkingMemoryManager,
    format_patient_card,
)
from .schemas import (
    ComplexityLevel,
    Episode,
    FinalAnswer,
    Plan,
    PlanStep,
    ToolResult,
    UserQuery,
)
from .tools.base import ToolRegistry
from .verifiers.l3_reflexion import GradedVerifierOrchestrator


class MedicalAgentOrchestrator:
    """
    医疗诊断 Agent 的主调度器
    
    用法:
        orch = MedicalAgentOrchestrator(planner=..., tool_registry=..., ...)
        answer = await orch.answer_async(UserQuery(text="头痛三天", user_id="u1"))
    """
    
    def __init__(
        self,
        planner: PlannerAgent,
        tool_registry: ToolRegistry,
        verifier_orchestrator: GradedVerifierOrchestrator,
        wm_manager: WorkingMemoryManager,
        episodic_memory: Optional[EpisodicMemory] = None,
        semantic_memory: Optional[SemanticMemory] = None,
        # 会话级 working memory 池（生产环境用 Redis）
        wm_pool: Optional[Dict[str, WorkingMemory]] = None,
        # —— 消融开关（真实生效，用于 eval/run_ablation.py）——
        enable_prefetch: bool = True,
        enable_memory_injection: bool = True,
        enable_ppr: bool = True,
        max_parallel_tools: int = 4,
        enable_memory_gating: bool = False,
        memory_gate_threshold: float = 0.2,
        max_verifier_level: str = "L3",   # "L1" | "L2" | "L3"，限制反思最高级别
    ):
        self.planner = planner
        self.tools = tool_registry
        self.verifier = verifier_orchestrator
        self.wm_manager = wm_manager
        self.episodic = episodic_memory
        self.semantic = semantic_memory
        self.wm_pool = wm_pool if wm_pool is not None else {}
        # 消融开关
        self.enable_prefetch = enable_prefetch
        self.enable_memory_injection = enable_memory_injection
        self.enable_ppr = enable_ppr
        self.max_parallel_tools = max_parallel_tools
        self.enable_memory_gating = enable_memory_gating
        self.memory_gate_threshold = memory_gate_threshold
        self.max_verifier_level = max_verifier_level
        # 把级别上限同步给 verifier 调度器（真实生效）
        if hasattr(self.verifier, "max_level"):
            self.verifier.max_level = max_verifier_level
        # 推测式预取的累计统计（用于演示命中率）
        self.prefetch_stats = {
            "total_prefetched": 0,   # 累计预取的候选数
            "total_hits": 0,         # 其中被后续真正需要并命中缓存的数
            "total_queries_with_prefetch": 0,
        }
        # 一次 answer 内的预取缓存：entity -> ToolResult
        self._prefetch_cache: Dict[str, ToolResult] = {}
        # 持有后台任务引用，避免被 GC 提前回收（asyncio 弱引用问题）
        self._background_tasks: set = set()
        self._background_errors: List[BaseException] = []
    
    # ============================================================
    # 主入口
    # ============================================================
    
    async def answer_async(self, user_query: UserQuery, session_id: Optional[str] = None) -> FinalAnswer:
        total_start = time.time()
        sid = session_id or user_query.query_id

        # 重置 LLM token 计数器（本次 answer 独立统计）
        self.planner.llm._token_count = 0
        
        # 1. 加载/创建 Working Memory
        wm = self.wm_pool.get(sid)
        critical_facts = {"allergies": [], "chronic_diseases": []}
        if wm is None:
            wm = self.wm_manager.create(session_id=sid, user_id=user_query.user_id)
            # 冷启动：从 Episodic 加载关键事实（过敏、慢性病）
            if self.enable_memory_injection and self.episodic and user_query.user_id:
                critical = self.episodic.retrieve_critical_facts(user_query.user_id)
                wm.patient_profile.allergies = critical.get("allergies", [])
                wm.patient_profile.medical_history = critical.get("chronic_diseases", [])
                critical_facts = {
                    "allergies": list(critical.get("allergies", [])),
                    "chronic_diseases": list(critical.get("chronic_diseases", [])),
                }
            self.wm_pool[sid] = wm

        # 关键事实进入 PatientProfile 后会跨当前 session 持续存在；每轮都如实上报。
        critical_facts = {
            "allergies": list(wm.patient_profile.allergies),
            "chronic_diseases": list(wm.patient_profile.medical_history),
        }
        
        self.wm_manager.add_user_turn(wm, user_query.text)
        
        # 2. 异步启动 Episodic 检索（不阻塞）
        episodic_task = None
        if self.enable_memory_injection and self.episodic and user_query.user_id:
            episodic_task = asyncio.create_task(
                asyncio.to_thread(
                    self.episodic.retrieve,
                    user_query.user_id, user_query.text, 5
                )
            )
        
        # 3. 等 Episodic 检索完成（实际并行的开销时间被 Planner 准备覆盖）
        episodic_hints = []
        if episodic_task:
            try:
                episodes = await episodic_task
                episodic_hints = [
                    {
                        "episode_id": e.episode_id,
                        "summary": e.summary,
                        "diagnoses": e.diagnoses,
                        "timestamp": e.timestamp.isoformat() if e.timestamp else None,
                        "status": e.status,
                        "provenance": e.provenance,
                        "retrieval_score": e.retrieval_score,
                        "retrieval_components": e.retrieval_components,
                    }
                    for e in episodes
                ]
            except Exception as e:
                episodic_hints = []

        # ── long-term memory gating ──
        gate_records: List[Dict] = []
        raw_episodic_hints = list(episodic_hints)
        raw_episodic_cnt = len(episodic_hints)
        raw_episodic_chars = len(str(episodic_hints)) if episodic_hints else 0
        if self.enable_memory_gating and episodic_hints:
            from src.memory.memory_gate import gate_episodic_hints
            episodic_hints, gate_records = gate_episodic_hints(
                user_query.text, episodic_hints,
                enabled=True, threshold=self.memory_gate_threshold,
            )

        # ── memory observability: 收集检索和上下文元数据 ──
        # working memory 始终参与 Planner（短期会话状态），不受 enable_memory_injection 控制。
        # enable_memory_injection 只控制 episodic/semantic long-term memory 注入。
        long_term_enabled = self.enable_memory_injection
        wm_context = self.wm_manager.get_planner_context(wm)
        wm_ctx_chars = len(str(wm_context)) if wm_context else 0
        critical_values = critical_facts["allergies"] + critical_facts["chronic_diseases"]
        episodic_cnt = len(episodic_hints)
        episodic_ctx_chars = len(str(episodic_hints)) if episodic_hints else 0
        filtered_cnt = raw_episodic_cnt - episodic_cnt
        lifecycle_counter = getattr(self.episodic, "lifecycle_counts", None)
        lifecycle_counts = (
            lifecycle_counter(user_query.user_id)
            if callable(lifecycle_counter) and user_query.user_id else {}
        )
        retrieved_episode_ids = [
            str(h.get("episode_id", "")) for h in raw_episodic_hints
            if h.get("episode_id")
        ]
        injected_episode_ids = [
            str(h.get("episode_id", "")) for h in episodic_hints
            if h.get("episode_id")
        ]
        long_term_chars = episodic_ctx_chars
        memory_meta = {
            "scope": {
                "user_id": user_query.user_id or "",
                "session_id": session_id or "",
                "memory_key": session_id or user_query.user_id or "anonymous",
            },
            "working": {
                "enabled": True,
                "context_chars": wm_ctx_chars,
                "turn_count": len(wm.conversation_history) if hasattr(wm, "conversation_history") else 0,
            },
            "critical": {
                "enabled": long_term_enabled and self.episodic is not None,
                "bypasses_gate": True,
                "allergies": critical_facts["allergies"],
                "chronic_diseases": critical_facts["chronic_diseases"],
                "injected_count": len(critical_values),
                "context_chars": len(str(critical_values)) if critical_values else 0,
            },
            "episodic": {
                "enabled": long_term_enabled and self.episodic is not None,
                "retrieved_count": raw_episodic_cnt,
                "injected_count": episodic_cnt,
                "filtered_count": filtered_cnt,
                "context_chars": episodic_ctx_chars,
                "raw_context_chars": raw_episodic_chars,
                "gating_enabled": self.enable_memory_gating,
                "gate_threshold": self.memory_gate_threshold,
                "gate_records": gate_records[:5],
                "retrieved_episode_ids": retrieved_episode_ids,
                "injected_episode_ids": injected_episode_ids,
                "embedding_model": getattr(self.episodic, "embedding_model_id", ""),
                "retrieval_mode": getattr(self.episodic, "retrieval_mode", ""),
                "retrieval_weights": list(getattr(
                    self.episodic, "retrieval_weights", (0.5, 0.3, 0.15, 0.05),
                )),
                "reranker_model": getattr(self.episodic, "reranker_model_id", ""),
                "reranker_candidate_k": getattr(self.episodic, "reranker_candidate_k", 0),
                "reranker_threshold": getattr(self.episodic, "reranker_threshold", None),
                "active_only": True,
                "lifecycle_counts": lifecycle_counts,
                "retrieval_records": [
                    {
                        "episode_id": hint.get("episode_id", ""),
                        "status": hint.get("status", "active"),
                        "provenance": hint.get("provenance", {}),
                        "score": hint.get("retrieval_score"),
                        "components": hint.get("retrieval_components", {}),
                    }
                    for hint in raw_episodic_hints[:5]
                ],
            },
            "semantic": {
                "enabled": False,
                "retrieved_count": 0,
                "context_chars": 0,
            },
            "injection": {
                "long_term_enabled": long_term_enabled,
                "long_term_context_chars": long_term_chars,
                "working_context_chars": wm_ctx_chars,
                "total_context_chars_with_working": wm_ctx_chars + long_term_chars,
                "total_retrieved_count": raw_episodic_cnt,
                "total_injected_count": episodic_cnt,
                "total_filtered_count": filtered_cnt,
            },
        }

        # 4. Planner 生成计划
        plan = self.planner.plan(
            query=user_query.text,
            working_memory=wm_context,
            episodic_hints=episodic_hints,
        )
        
        # 5. 应用 Planner 的 Working Memory 更新
        if plan.memory_update:
            self.wm_manager.apply_planner_update(wm, plan.memory_update)
        
        # 6. 执行计划（DAG 并行调度）
        #    先后台启动推测式预取，再跑主 DAG，主 DAG 跑完时回收预取
        prefetch_task = self._launch_prefetch(plan)
        execution_results, execution_meta = await self._execute_plan_with_meta(
            plan, query=user_query.text, prefetch_task=prefetch_task
        )
        
        # 把工具结果加入 Working Memory 的证据池
        for step in plan.steps:
            if step.output and isinstance(step.output, ToolResult) and step.output.success:
                if isinstance(step.output.data, dict):
                    self.wm_manager.add_tool_evidence(wm, step.tool, step.output.data)
        
        # 7. 合成 draft answer
        draft = self._synthesize_draft(user_query.text, plan, execution_results)

        # Guard: _synthesize_draft 应始终返回 dict，但防御间歇性异常
        if not isinstance(draft, dict):
            draft = {"content": str(draft) if draft else "", "citations": []}

        # 8. 分级反思
        verify_output = self.verifier.verify(
            query=user_query.text,
            draft_answer=draft,
            evidence=draft.get("citations", []),
            complexity=plan.complexity.value,
            patient_profile=wm.patient_profile,
        )
        
        # Replan 兜底
        if verify_output.get("needs_replan"):
            new_plan = self.planner.replan(
                original_query=user_query.text,
                previous_plan=plan,
                failure_hint="; ".join(verify_output["verify_chain"][-1].errors),
                working_memory=self.wm_manager.get_planner_context(wm),
            )
            new_prefetch_task = self._launch_prefetch(new_plan)
            new_results = await self._execute_plan(
                new_plan, query=user_query.text, prefetch_task=new_prefetch_task
            )
            draft = self._synthesize_draft(user_query.text, new_plan, new_results)
            if not isinstance(draft, dict):
                draft = {"content": str(draft) if draft else "", "citations": []}
            verify_output = self.verifier.verify(
                query=user_query.text,
                draft_answer=draft,
                evidence=draft.get("citations", []),
                complexity=new_plan.complexity.value,
                patient_profile=wm.patient_profile,
            )
            plan = new_plan
        
        final_dict = verify_output["final_answer"]

        # Guard: L3 reflexion or transient LLM output may return a string instead of dict
        if not isinstance(final_dict, dict):
            final_dict = {"content": str(final_dict) if final_dict else "", "citations": []}

        # 9. 构造最终答案
        total_elapsed_ms = (time.time() - total_start) * 1000
        final_answer = FinalAnswer(
            content=final_dict.get("content", ""),
            diagnoses=final_dict.get("possible_diagnoses") or plan.diagnoses,
            citations=final_dict.get("citations", []),
            reasoning_trace=[s.tool + ": " + str(s.output)[:200]
                             for s in plan.steps if s.output],
            confidence=self._compute_confidence(verify_output),
            plan=plan,
            verification_results=verify_output["verify_chain"],
            verification_meta={
                "level_reached": verify_output.get("level_reached", ""),
                "trigger_reason": verify_output.get("trigger_reason", ""),
                "l3_skip_reason": verify_output.get("l3_skip_reason", ""),
                "needs_replan": verify_output.get("needs_replan", False),
            },
            execution_meta=execution_meta,
            memory_meta=memory_meta,
            total_elapsed_ms=total_elapsed_ms,
            total_tokens=self.planner.llm._token_count,
        )
        
        # 10. 加入 Working Memory + 异步写入 Episodic
        self.wm_manager.add_assistant_turn(wm, final_answer.content)
        if self.episodic and user_query.user_id:
            bg = asyncio.create_task(self._async_write_episode(
                user_query, final_answer, wm,
                l3_triggered=(verify_output["level_reached"] == "L3"),
            ))
            self._background_tasks.add(bg)
            bg.add_done_callback(self._on_background_task_done)
        
        return final_answer

    async def flush_memory_writes(self) -> None:
        """等待当前已提交的后台记忆写入完成，供多轮评测和有序关停使用。"""
        while self._background_tasks:
            tasks = list(self._background_tasks)
            await asyncio.gather(*tasks, return_exceptions=True)
        if self._background_errors:
            errors = self._background_errors[:]
            self._background_errors.clear()
            raise RuntimeError(f"episodic memory write failed: {errors[0]}") from errors[0]

    def _on_background_task_done(self, task: "asyncio.Task") -> None:
        self._background_tasks.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            self._background_errors.append(error)
    
    def answer(self, user_query: UserQuery, session_id: Optional[str] = None) -> FinalAnswer:
        """同步接口"""
        return asyncio.run(self.answer_async(user_query, session_id))
    
    # ============================================================
    # 内部方法
    # ============================================================
    
    async def _execute_plan(
        self,
        plan: Plan,
        query: str = "",
        prefetch_task: Optional["asyncio.Task"] = None,
    ) -> Dict[int, ToolResult]:
        """DAG 调度 wrapper：调用 _execute_plan_with_meta 并只返回 results。"""
        results, _ = await self._execute_plan_with_meta(plan, query, prefetch_task)
        return results

    async def _execute_plan_with_meta(
        self,
        plan: Plan,
        query: str = "",
        prefetch_task: Optional["asyncio.Task"] = None,
    ) -> Tuple[Dict[int, ToolResult], Dict[str, Any]]:
        """
        DAG 调度：拓扑排序后分层并行执行（含执行元数据）。

        在 _execute_plan 基础上增加可观测性：记录 layer widths、wall time、
        parallelism ratio（sum tool elapsed / wall time）、max layer width 等。

        Returns:
            (results, execution_meta)
        """
        tool_wall_start = time.time()
        results: Dict[int, ToolResult] = {}
        completed = set()
        layer_widths: List[int] = []

        # 消融：PPR OFF — 从执行计划中移除 ppr_reasoner 步骤
        if not self.enable_ppr:
            removed_ids = {s.step_id for s in plan.steps if s.tool == "ppr_reasoner"}
            plan.steps = [s for s in plan.steps if s.tool != "ppr_reasoner"]
            for s in plan.steps:
                s.depends_on = [d for d in s.depends_on if d not in removed_ids]

        # 局部 semaphore：每次执行独立创建，避免跨 event loop 绑定风险
        semaphore = asyncio.Semaphore(self.max_parallel_tools)

        while len(completed) < len(plan.steps):
            ready_steps = [
                s for s in plan.steps
                if s.step_id not in completed
                and all(dep in completed for dep in s.depends_on)
            ]

            if not ready_steps:
                for s in plan.steps:
                    if s.step_id not in completed:
                        results[s.step_id] = ToolResult(
                            tool_name=s.tool, success=False, error="dependency_deadlock"
                        )
                break

            layer_widths.append(len(ready_steps))

            async def _run_with_semaphore(step, tool, resolved_input):
                async with semaphore:
                    return await self._run_step(step, tool, resolved_input)

            tasks = []
            for step in ready_steps:
                resolved_input = self._resolve_input(step.input, results, query)
                tool = self.tools.get(step.tool)
                tasks.append(_run_with_semaphore(step, tool, resolved_input))

            layer_results = await asyncio.gather(*tasks, return_exceptions=True)

            for step, result in zip(ready_steps, layer_results):
                if isinstance(result, Exception):
                    result = ToolResult(
                        tool_name=step.tool, success=False, error=str(result)
                    )
                results[step.step_id] = result
                step.output = result
                step.status = "done" if result.success else "failed"
                step.elapsed_ms = result.elapsed_ms
                completed.add(step.step_id)

        tool_wall_ms = (time.time() - tool_wall_start) * 1000

        # 回收推测式预取，记录命中情况
        if prefetch_task is not None:
            await self._reconcile_prefetch(prefetch_task, plan, results)

        # 构建执行元数据
        tool_sum_elapsed_ms = sum(
            r.elapsed_ms for r in results.values() if r.elapsed_ms is not None
        )
        parallelism_ratio = (
            tool_sum_elapsed_ms / tool_wall_ms if tool_wall_ms > 0 else 0.0
        )
        execution_meta = {
            "tool_wall_ms": round(tool_wall_ms, 1),
            "tool_sum_elapsed_ms": round(tool_sum_elapsed_ms, 1),
            "parallelism_ratio": round(parallelism_ratio, 2),
            "layer_count": len(layer_widths),
            "max_layer_width": max(layer_widths) if layer_widths else 0,
            "layer_widths": layer_widths,
            "executed_step_count": len(results),
            "max_parallel_tools": self.max_parallel_tools,
        }

        return results, execution_meta
    
    async def _run_step(self, step: PlanStep, tool, resolved_input) -> ToolResult:
        step.status = "running"
        return await tool.ainvoke(resolved_input)
    
    # ============================================================
    # 推测式预取 (Speculative Pre-fetch)
    # ============================================================
    
    def _launch_prefetch(self, plan: Plan) -> Optional["asyncio.Task"]:
        """
        根据 plan.speculative_prefetch 后台并行预取候选实体的 KG 资料。

        设计动机:
        - Planner 在生成计划时，会顺手列出"可能相关但还不确定"的候选诊断
          （如鉴别诊断时列出脑膜炎、流感、偏头痛）。
        - 主 DAG 跑的同时，后台用 kg_local_search 把这些候选的邻居资料先拉好。
        - 等主链路（如 PPR / global search）真正确定了高相关诊断，
          若它恰好在预取候选里，就能直接复用，省掉一次串行 KG 查询。

        命中率取决于 Planner 的预判质量，实测约 45%（见 docs/02）。
        返回一个后台 Task；若无候选则返回 None。
        """
        if not self.enable_prefetch:
            return None
        candidates = plan.speculative_prefetch or []
        if not candidates:
            return None
        if "kg_local_search" not in self.tools.list_names():
            return None
        
        self._prefetch_cache = {}
        
        async def _do_prefetch():
            tool = self.tools.get("kg_local_search")
            # 并行预取所有候选
            async def fetch_one(entity):
                res = await tool.ainvoke({"query": entity, "entities": [entity]})
                return entity, res
            pairs = await asyncio.gather(
                *[fetch_one(e) for e in candidates],
                return_exceptions=True,
            )
            cache = {}
            for pair in pairs:
                if isinstance(pair, Exception):
                    continue
                entity, res = pair
                # 只缓存真正找到了 KG 数据的实体（facts 非空），
                # 避免无效实体膨胀 total_prefetched
                if res.success and isinstance(res.data, dict) and res.data.get("facts"):
                    cache[entity] = res
            return cache
        
        return asyncio.create_task(_do_prefetch())
    
    async def _reconcile_prefetch(
        self,
        prefetch_task: "asyncio.Task",
        plan: Plan,
        results: Dict[int, ToolResult],
    ) -> None:
        """
        主 DAG 执行完成后，回收预取结果并统计命中。

        命中定义: 主链路实际产出的诊断/概念中，有任意一个在预取候选里，
        且该候选的预取结果成功 —— 说明这次预取"赚到了"。
        命中的预取结果存入 self._prefetch_cache，供 _synthesize_draft 复用。
        """
        try:
            cache = await prefetch_task
        except Exception:
            cache = {}
        
        if not cache:
            return
        
        # 从主链路结果中收集实际涉及的实体（诊断/概念）
        # 扩展提取：除了 facts/relevant_concepts/candidate_entities，
        # 还包括 NER 提取的实体、KG 查询的实体、PPR 种子实体
        produced_entities = set()
        for res in results.values():
            if not res.success or not isinstance(res.data, dict):
                continue
            data = res.data
            # KG local search: facts 中的目标实体 + 查询过的实体
            for fact in data.get("facts", []):
                if fact.get("target"):
                    produced_entities.add(fact["target"])
                if fact.get("source"):
                    produced_entities.add(fact["source"])
            for ent in data.get("entities_queried", []):
                produced_entities.add(ent)
            # PPR: 种子实体 + 扩散概念
            for ent in data.get("seed_entities", []):
                produced_entities.add(ent)
            for item in data.get("relevant_concepts", []):
                if item.get("entity"):
                    produced_entities.add(item["entity"])
            # KG global search: 候选实体 + 社区核心实体
            for ent in data.get("candidate_entities", []):
                produced_entities.add(ent)
            for c in data.get("communities", []):
                for ent in c.get("core_entities", []):
                    produced_entities.add(ent)
            # NER: 提取到的所有实体
            for ent in data.get("all_entities", []):
                produced_entities.add(ent)
        
        # 统计命中
        prefetched = list(cache.keys())
        hits = [e for e in prefetched if e in produced_entities]
        
        self.prefetch_stats["total_prefetched"] += len(prefetched)
        self.prefetch_stats["total_hits"] += len(hits)
        self.prefetch_stats["total_queries_with_prefetch"] += 1
        
        # 命中的预取结果留给 synthesize 复用
        self._prefetch_cache = {e: cache[e] for e in hits}
    
    def get_prefetch_hit_rate(self) -> float:
        """累计预取命中率（命中候选数 / 预取候选数）"""
        total = self.prefetch_stats["total_prefetched"]
        if total == 0:
            return 0.0
        return self.prefetch_stats["total_hits"] / total
    
    def _resolve_input(
        self,
        raw_input: Any,
        results: Dict[int, ToolResult],
        query: str = "",
    ) -> Any:
        """
        解析输入中的引用占位符。

        支持的引用格式:
        - ${query}                     原始用户查询
        - ${stepN.output}              引用 stepN 的完整 output（保留原始类型）
        - ${stepN.output.field}        引用 output 中的字段
        - ${stepN.output.a.b}          多层嵌套字段

        关键设计:
        1. 若整个字符串就是单个占位符（如 input="${step1.output.all_entities}"），
           直接返回被引用对象的**原始类型**（list / dict / int...），不强转字符串。
           这样 ppr_reasoner 的 seed_entities 能拿到真正的 list 而非 "[...]" 字符串。
        2. 若占位符内嵌在更长的字符串里（如 "症状: ${...}"），才做字符串替换。
        """
        import re

        REF_PATTERN = re.compile(r"\$\{([^}]+)\}")

        def lookup(ref: str):
            """解析单个引用表达式，返回 (值, 是否命中)"""
            ref = ref.strip()
            if ref == "query":
                return query, True
            step_match = re.match(r"step(\d+)\.output((?:\.\w+)*)$", ref)
            if not step_match:
                return None, False
            step_id = int(step_match.group(1))
            field_path = step_match.group(2)  # 形如 ".all_entities" 或 ".a.b" 或 ""
            res = results.get(step_id)
            if not res or not res.success:
                return None, False
            value = res.data
            for field in [f for f in field_path.split(".") if f]:
                if isinstance(value, dict):
                    value = value.get(field)
                else:
                    return None, False
            return value, True

        if isinstance(raw_input, str):
            # 情况 A：整个字符串就是单个占位符 -> 返回原始类型
            full_match = REF_PATTERN.fullmatch(raw_input.strip())
            if full_match:
                value, hit = lookup(full_match.group(1))
                return value if hit else raw_input
            # 情况 B：占位符内嵌在文本中 -> 字符串替换
            def replace_match(m):
                value, hit = lookup(m.group(1))
                return str(value) if hit else m.group(0)
            return REF_PATTERN.sub(replace_match, raw_input)

        if isinstance(raw_input, dict):
            return {k: self._resolve_input(v, results, query) for k, v in raw_input.items()}
        if isinstance(raw_input, list):
            return [self._resolve_input(v, results, query) for v in raw_input]
        return raw_input
    
    def _synthesize_draft(
        self,
        query: str,
        plan: Plan,
        results: Dict[int, ToolResult],
    ) -> Dict[str, Any]:
        """
        合成 draft answer

        策略：以 Planner 的 thought（诊断推理）为主体，
        工具结果（KG 事实、社区摘要、PPR 概念）作为补充证据。
        """
        # 收集所有 successful 工具的 citations
        all_citations = []
        all_diagnoses = []
        all_drugs = []
        community_summaries = []

        # 主链路结果 + 命中的预取结果（预取命中即可零成本复用其 KG 资料）
        all_results = list(results.values()) + list(self._prefetch_cache.values())
        for res in all_results:
            if not res.success or not isinstance(res.data, dict):
                continue
            all_citations.extend(res.citations)
            data = res.data

            # 提取潜在诊断
            for fact in data.get("facts", []):
                if fact.get("target_type") == "disease":
                    all_diagnoses.append(fact["target"])
                elif fact.get("target_type") == "drug":
                    all_drugs.append(fact["target"])

            # 社区摘要
            for c in data.get("communities", []):
                community_summaries.append(c.get("summary", ""))

            # PPR 结果
            for item in data.get("relevant_concepts", [])[:5]:
                all_diagnoses.append(item.get("entity"))

        # 去重
        all_diagnoses = list(dict.fromkeys(all_diagnoses))[:5]
        all_drugs = list(dict.fromkeys(all_drugs))[:3]

        # 合成回答：以 Planner thought 为主体
        content_parts = [f"针对您的问题「{query}」，基于知识图谱和检索证据：\n"]

        # Planner 的诊断推理（核心内容）
        thought = plan.thought or ""
        if thought:
            content_parts.append(thought)
            content_parts.append("")

        # 工具检索到的补充证据（有则追加）
        if community_summaries:
            content_parts.append("**补充 - 相关知识背景**:")
            for s in community_summaries[:2]:
                content_parts.append(f"- {s}")
            content_parts.append("")

        content_parts.append("\n⚠️ 此回答基于知识库推理，仅供参考，请以专业医生诊断为准。")

        # citations 构建：验证 thought 引用 + 合并真实 KG 数据
        # 策略：真实 KG facts → verified，thought 中匹配的 → verified，
        #       thought 中未匹配的 → inferred（标注来源）
        verified_citations = []
        inferred_citations = []

        # 1. 工具返回的真实 KG citations（最高可信度）
        verified_citations.extend(all_citations)

        # 2. 构建真实 KG 事实集合，用于验证 thought 引用
        real_kg_facts = set()
        for res in all_results:
            if not res.success or not isinstance(res.data, dict):
                continue
            for fact in res.data.get("facts", []):
                src = fact.get("source", "")
                rel = fact.get("rel", "")
                dst = fact.get("target", "")
                if src and rel and dst:
                    real_kg_facts.add((src, rel, dst))

        # 3. 从 thought 提取引用，验证后分类
        if thought:
            import re
            kg_refs = re.findall(r"(\S+)\s*-\[(.+?)\]->\s*(\S+)", thought)
            for src, rel, dst in kg_refs:
                triple = (src, rel, dst)
                if triple in real_kg_facts:
                    # 已在 verified 中（来自工具），跳过避免重复
                    if not any(c.get("source") == src and c.get("rel") == rel and c.get("target") == dst
                               for c in verified_citations):
                        verified_citations.append({"type": "kg_fact", "source": src, "rel": rel, "target": dst})
                else:
                    inferred_citations.append({"type": "kg_inferred", "source": src, "rel": rel, "target": dst})

        all_citations = verified_citations + inferred_citations

        return {
            "content": "\n".join(content_parts),
            "possible_diagnoses": all_diagnoses,
            "recommended_drugs": all_drugs,
            "citations": all_citations,
        }
    
    def _compute_confidence(self, verify_output: Dict) -> float:
        chain = verify_output.get("verify_chain", [])
        if not chain:
            return 0.5
        last = chain[-1]
        # scores 可能混入非数值字段（如 L2 的 "issues" 列表），只取数值维度
        numeric_scores = [
            v for v in (last.scores or {}).values()
            if isinstance(v, (int, float))
        ]
        if numeric_scores:
            return min(numeric_scores)
        return 0.85 if last.passed else 0.5
    
    async def _async_write_episode(
        self,
        user_query: UserQuery,
        final_answer: FinalAnswer,
        wm: WorkingMemory,
        l3_triggered: bool = False,
    ) -> None:
        """异步写入 Episodic Memory（不阻塞主流程）"""
        if not self.episodic:
            return
        episode = Episode(
            user_id=user_query.user_id,
            timestamp=user_query.timestamp,
            episode_type="consultation",
            diagnoses=list(final_answer.diagnoses or []),
            symptoms=[
                s.get("name") if isinstance(s, dict) else str(s)
                for s in wm.patient_profile.symptoms
            ],
            summary=f"Q: {user_query.text[:100]} | A: {final_answer.content[:200]}",
            provenance={
                "source_type": "agent_consultation",
                "source_id": user_query.query_id,
                "recorded_by": "medical_agent",
            },
        )
        self.episodic.write(episode, l3_triggered=l3_triggered)
