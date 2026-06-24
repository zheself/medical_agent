"""
demo/demo_full_flow.py — 完整流程演示

运行：
    cd medical_agent
    python -m demo.demo_full_flow

这个 demo 在没有 GPU、没有 Neo4j、没有 vLLM 的情况下能完整跑通整个流程，
让你看到每一步的输出，方便理解架构。

切换到真实模型/服务的步骤详见 docs/01_architecture.md
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

# 添加 src 到 path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agents.planner import MockLLMBackend, PlannerAgent
from src.memory.episodic_memory import (
    EpisodicMemory,
    Episode,
    ImportanceScorer,
    MockEmbedder,
    SQLiteEpisodicBackend,
)
from src.memory.semantic_memory import SemanticMemory
from src.memory.working_memory import MockSummarizer, WorkingMemoryManager
from src.orchestrator import MedicalAgentOrchestrator
from src.schemas import UserQuery
from src.tools.base import ToolRegistry
from src.tools.kg_global_search import KGGlobalSearchTool, MockCommunityVectorStore
from src.tools.kg_global_search import MockEmbedder as VEmbedder
from src.tools.kg_local_search import KGLocalSearchTool, MockKGBackend
from src.tools.ner_tool import NERTool
from src.tools.ppr_reasoner import PPRReasonerTool
from src.verifiers.l1_rule_verifier import L1RuleVerifier
from src.verifiers.l2_model_verifier import L2ModelVerifier, MockSmallModel
from src.verifiers.l3_reflexion import GradedVerifierOrchestrator, L3Reflexion


def build_system(backend: str = "mock", db_path: str = None):
    """组装完整系统。委托给 src.factory。

    backend="mock": 全内存 Mock 后端（默认，零依赖）
    backend="db":   从 SQLite .db 读取（需先跑 scripts/seed_database.py）
    """
    from src.factory import build_system as _build
    return _build(backend=backend, db_path=db_path)


def print_section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


async def demo(backend="mock", db_path=None):
    print_section("STEP 1: 组装系统")
    orch, episodic = build_system(backend=backend, db_path=db_path)
    mode_desc = {"mock": "全内存 Mock", "db": "从 SQLite .db 读取真实数据", "vllm": "真实 LLM（vLLM）"}
    print(f"✅ 系统组装完成，后端模式: {backend}（{mode_desc.get(backend, '')}）")
    print(f"   - 已注册工具: {orch.tools.list_names()}")
    
    # 预先写一些 user u1 的历史，模拟"老用户"场景
    print_section("STEP 2: 预置用户历史（模拟老用户）")
    past_episodes = [
        Episode(
            user_id="u1",
            episode_type="diagnosis",
            diagnoses=["糖尿病"],
            medications=["二甲双胍"],
            symptoms=["多饮", "多尿"],
            summary="2024-08-15 确诊2型糖尿病，开始服用二甲双胍",
        ),
        Episode(
            user_id="u1",
            episode_type="consultation",
            diagnoses=["偏头痛"],
            medications=["布洛芬"],
            symptoms=["头痛"],
            summary="2024-12-15 偶发偏头痛，建议布洛芬缓解，提示青霉素过敏",
        ),
    ]
    for ep in past_episodes:
        written = episodic.write(ep)
        print(f"   episode '{ep.summary[:40]}...' → 写入={written}, importance={ep.importance_score:.2f}")
    
    # ============================================================
    # Case 1: 鉴别诊断类查询（高复杂度）
    # ============================================================
    print_section("STEP 3: Case 1 — 鉴别诊断查询")
    query1 = UserQuery(
        user_id="u1",
        text="我最近三天持续头痛伴发烧 38.5°C，颈部僵硬，可能是什么病？"
    )
    print(f"用户输入: {query1.text}")
    
    answer1 = await orch.answer_async(query1)
    print(f"\n📌 Planner 计划:")
    print(f"   思考: {answer1.plan.thought}")
    print(f"   复杂度: {answer1.plan.complexity.value}")
    print(f"   步骤: {[(s.step_id, s.tool) for s in answer1.plan.steps]}")
    
    print(f"\n📌 验证链路:")
    for vr in answer1.verification_results:
        status = "✅" if vr.passed else "❌"
        print(f"   {status} {vr.level.value}: {vr.elapsed_ms:.1f}ms, errors={vr.errors}")
    
    print(f"\n📌 最终回答:")
    print(answer1.content)
    print(f"\n📊 引用数: {len(answer1.citations)}, 总耗时: {answer1.total_elapsed_ms:.1f}ms")
    
    # ============================================================
    # Case 2: 简单事实查询（低复杂度）
    # ============================================================
    print_section("STEP 4: Case 2 — 简单事实查询")
    query2 = UserQuery(
        user_id="u1",
        text="二甲双胍是什么药？"
    )
    print(f"用户输入: {query2.text}")
    answer2 = await orch.answer_async(query2)
    print(f"\n📌 复杂度: {answer2.plan.complexity.value}")
    print(f"📌 步骤数: {len(answer2.plan.steps)}")
    print(f"📌 验证级别: {[vr.level.value for vr in answer2.verification_results]}")
    print(f"📌 回答:\n{answer2.content[:300]}")
    print(f"📊 总耗时: {answer2.total_elapsed_ms:.1f}ms")
    
    # ============================================================
    # Case 3: 多轮对话（验证 Working Memory）
    # ============================================================
    print_section("STEP 5: Case 3 — 多轮对话（验证记忆）")
    
    # 同一 session
    sid = "session_multi_turn"
    
    q3a = UserQuery(user_id="u2", text="我有头痛，从昨天开始")
    a3a = await orch.answer_async(q3a, session_id=sid)
    print(f"轮次1: {q3a.text}")
    print(f"  → 回答片段: {a3a.content[:100]}...")
    
    q3b = UserQuery(user_id="u2", text="现在又开始发烧了")
    a3b = await orch.answer_async(q3b, session_id=sid)
    print(f"\n轮次2: {q3b.text}")
    print(f"  → 回答片段: {a3b.content[:100]}...")
    
    # 检查 Working Memory 是否记住了上一轮的症状
    wm = orch.wm_pool.get(sid)
    if wm:
        from src.memory.working_memory import format_patient_card
        print(f"\n📋 Working Memory 累积的患者档案:")
        print(format_patient_card(wm.patient_profile))
    
    # ============================================================
    # Case 4: PPR 多跳推理单独演示
    # ============================================================
    print_section("STEP 6: Case 4 — Personalized PageRank 多跳推理")
    
    ppr_tool = orch.tools.get("ppr_reasoner")
    ppr_result = await ppr_tool.ainvoke({
        "seed_entities": ["头痛", "发烧", "颈强直"],
        "top_k": 10,
        "alpha": 0.5,
    })
    print(f"种子实体: {ppr_result.data['seed_entities']}")
    print(f"alpha: {ppr_result.data['alpha']}")
    print(f"\n扩散得到的相关概念 (top 10):")
    for item in ppr_result.data["relevant_concepts"]:
        print(f"   {item['entity']:15s} → PPR 分数: {item['ppr_score']:.4f}")
    
    print(f"\n💡 解读: 系统不依赖固定路径，自动'联想'出脑膜炎为最相关疾病")
    print(f"   alpha=0.5 vs 默认 0.85 的对比效果，详见 docs/05_graphrag.md")
    
    # ============================================================
    # Case 5: L1 规则触发演示
    # ============================================================
    print_section("STEP 7: Case 5 — L1 规则反思触发")
    
    from src.schemas import PatientProfile
    from src.verifiers.l1_rule_verifier import L1RuleVerifier
    
    l1 = L1RuleVerifier()
    
    # 构造一个"违规"的回答：青霉素过敏患者推荐青霉素
    bad_answer = {
        "content": "建议使用青霉素治疗",
        "recommended_drugs": ["青霉素"],
        "possible_diagnoses": ["细菌感染"],
        "citations": [{"type": "kg_fact", "source": "demo"}],
    }
    profile = PatientProfile(age=30, gender="男", allergies=["青霉素"])
    
    result = l1.verify(bad_answer, profile)
    print(f"输入: 给青霉素过敏患者推荐青霉素")
    print(f"L1 校验: {'✅ 通过' if result.passed else '❌ 拦截'}")
    print(f"耗时: {result.elapsed_ms:.2f}ms  (注意: 0 LLM 调用)")
    for err in result.errors:
        print(f"  错误: {err}")
    
    # ============================================================
    # Case 6: GraphRAG 社区检测 + 摘要生成
    # ============================================================
    print_section("STEP 8: Case 6 — GraphRAG 社区检测 + 回填验证")
    from src.graphrag import CommunityDetector, CommunitySummaryGenerator
    from src.tools.kg_local_search import MockKGBackend
    kg = MockKGBackend()
    edges = []
    for s, facts in kg.MOCK_FACTS.items():
        for f in facts:
            edges.append((s, f["target"], f.get("weight", 0.5)))
    detector = CommunityDetector(edges)
    levels = detector.detect_hierarchical(resolutions=[0.5, 1.0, 1.5])
    print(f"  社区分层: l0={detector.num_communities(levels['l0'])} / "
          f"l1={detector.num_communities(levels['l1'])} / "
          f"l2={detector.num_communities(levels['l2'])} 个社区 (粗->细)")
    members = detector.community_members(levels["l1"])
    gen = CommunitySummaryGenerator()
    for cid, nodes in sorted(members.items()):
        if len(nodes) < 3:
            continue
        rels = [{"src": s, "rel": f["rel"], "dst": f["target"]}
                for s, facts in kg.MOCK_FACTS.items() if s in set(nodes)
                for f in facts if f["target"] in set(nodes)]
        r = gen.generate(f"L1_C{cid:03d}", 1, nodes, rels)
        flag = "⚠️幻觉" if r.is_hallucinated else "✅"
        print(f"    [{r.community_id}] {r.theme}: {nodes}  {flag}(ratio={r.hallucination_ratio:.2f})")
    print("  💡 回填验证: 摘要中的实体必须在原社区内，虚构率>10%则重新生成")

    # ============================================================
    # 推测式预取统计
    # ============================================================
    print_section("STEP 9: Speculative Pre-fetch 统计")
    print(f"  累计预取候选: {orch.prefetch_stats['total_prefetched']}")
    print(f"  命中: {orch.prefetch_stats['total_hits']}")
    print(f"  命中率: {orch.get_prefetch_hit_rate():.0%}")
    print("  💡 注: Mock 下命中率偏高（候选与主链路高度重合）；")
    print("     真实场景 Planner 预判质量有限，文档报告约 45%")
    
    # ============================================================
    # 总结
    # ============================================================
    print_section("✅ Demo 完成")
    print("""
关键观察:
1. Planner 根据复杂度自适应生成 DAG 计划，简单查询用更少步骤
2. 多轮对话中 Working Memory 累积患者档案（症状从1个变2个）
3. PPR 在没有固定路径模板的情况下，正确识别出脑膜炎为高相关
4. L1 规则反思 0 LLM 调用，毫秒级拦截药物过敏冲突
5. 整个流程无需 GPU、无需外部服务

切换到生产环境的步骤:
- MockLLMBackend → vLLM/API
- MockKGBackend → Neo4j
- MockCommunityVectorStore → Zilliz/Milvus  
- MockSmallModel → 蒸馏的 1.5B Verifier
详见 docs/01_architecture.md
""")


if __name__ == "__main__":
    import argparse
    _p = argparse.ArgumentParser()
    _p.add_argument("--db", action="store_true", help="使用 SQLite .db 后端（需先跑 scripts/seed_database.py）")
    _p.add_argument("--db-path", default=None, help="指定 .db 路径")
    _p.add_argument("--backend", choices=["mock", "db", "vllm"], default=None,
                    help="后端模式: mock（默认）, db（SQLite）, vllm（真实 LLM）")
    _args = _p.parse_args()
    if _args.backend:
        _backend = _args.backend
    else:
        _backend = "db" if (_args.db or _args.db_path) else "mock"
    asyncio.run(demo(backend=_backend, db_path=_args.db_path))
