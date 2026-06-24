"""
factory.py — 系统构建工厂

统一的 Agent 系统组装入口，支持两种后端模式:

- build_system(backend="mock")
    全内存 Mock 后端。KG 是硬编码 dict，Episodic 用 :memory:。
    零依赖、零准备，适合纯逻辑演示和单测。

- build_system(backend="db", db_path="data/medical_agent.db")
    "连真库"模式。KG / Episodic / Semantic 都从 SQLite .db 读取。
    需要先运行 scripts/seed_database.py 生成数据库。
    更接近真实部署形态（再往上换成 vLLM/Neo4j/Milvus 即生产）。

两种模式返回完全一致的 (orchestrator, episodic) 接口，可无缝切换。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from .agents.planner import MockLLMBackend, PlannerAgent
from .memory.episodic_memory import (
    EpisodicMemory, ImportanceScorer, MockEmbedder, SQLiteEpisodicBackend,
)
from .memory.semantic_memory import (
    DynamicRuleStore, FailureCaseStore, SemanticMemory,
)
from .memory.working_memory import MockSummarizer, WorkingMemoryManager
from .orchestrator import MedicalAgentOrchestrator
from .tools.base import ToolRegistry
from .tools.kg_global_search import (
    KGGlobalSearchTool, MockCommunityVectorStore, MockEmbedder as VEmbedder,
)
from .tools.kg_local_search import (
    KGLocalSearchTool, MockKGBackend, SQLiteKGBackend,
)
from .tools.ner_tool import NERTool
from .tools.ppr_reasoner import PPRReasonerTool
from .verifiers.l1_rule_verifier import L1RuleVerifier
from .verifiers.l2_model_verifier import L2ModelVerifier, MockSmallModel
from .verifiers.l3_reflexion import GradedVerifierOrchestrator, L3Reflexion


DEFAULT_DB = str(Path(__file__).resolve().parent.parent / "data" / "medical_agent.db")


def build_system(
    backend: str = "mock",
    db_path: Optional[str] = None,
    alpha: float = 0.5,
) -> Tuple[MedicalAgentOrchestrator, EpisodicMemory]:
    """
    组装完整 Agent 系统。

    Args:
        backend: "mock"（全内存）或 "db"（SQLite .db）
        db_path: backend="db" 时的数据库路径，默认 data/medical_agent.db
        alpha: PPR 的 alpha 参数

    Returns:
        (orchestrator, episodic_memory)
    """
    if backend not in ("mock", "db"):
        raise ValueError(f"backend 必须是 'mock' 或 'db'，得到 {backend}")

    use_db = backend == "db"
    db_path = db_path or DEFAULT_DB

    # ---------- KG 后端 ----------
    if use_db:
        kg_backend = SQLiteKGBackend(db_path)
    else:
        kg_backend = MockKGBackend()

    # ---------- 向量库（GraphRAG Global Search）----------
    # 注：当前 Global Search 走关键词匹配路径，mock/db 共用 MockCommunityVectorStore。
    # db 模式下社区数据已在 .db 的 communities 表，可按需扩展 SQLiteCommunityVectorStore。
    vector_store = MockCommunityVectorStore()
    embedder = VEmbedder()

    # ---------- Tools ----------
    tool_registry = ToolRegistry()
    tool_registry.register(NERTool())
    tool_registry.register(KGLocalSearchTool(kg_backend=kg_backend))
    tool_registry.register(KGGlobalSearchTool(vector_store=vector_store, embedder=embedder))
    tool_registry.register(PPRReasonerTool(kg_backend=kg_backend, alpha=alpha))

    # ---------- Planner ----------
    llm = MockLLMBackend()
    planner = PlannerAgent(
        llm=llm,
        tool_descriptions=tool_registry.get_all_descriptions(),
        enable_self_critique=True,
    )

    # ---------- Memory ----------
    wm_manager = WorkingMemoryManager(summarizer=MockSummarizer())
    epi_embedder = MockEmbedder()

    episodic_db = db_path if use_db else ":memory:"
    episodic = EpisodicMemory(
        backend=SQLiteEpisodicBackend(episodic_db),
        embedder=epi_embedder,
        scorer=ImportanceScorer(),
    )

    # Semantic：db 模式下失败案例/动态规则也指向同一个 .db
    if use_db:
        semantic = SemanticMemory(
            kg_backend=kg_backend,
            failure_store=FailureCaseStore(db_path),
            rule_store=_load_dynamic_rules_from_db(db_path),
            embedder=epi_embedder,
        )
    else:
        semantic = SemanticMemory(kg_backend=kg_backend, embedder=epi_embedder)

    # ---------- Verifiers ----------
    l1 = L1RuleVerifier(dynamic_rule_store=semantic.rule_store)
    l2 = L2ModelVerifier(small_model=MockSmallModel(), threshold=0.7)
    l3 = L3Reflexion(llm_backend=llm, semantic_memory=semantic)
    verifier_orch = GradedVerifierOrchestrator(l1, l2, l3)

    # ---------- Orchestrator ----------
    orch = MedicalAgentOrchestrator(
        planner=planner,
        tool_registry=tool_registry,
        verifier_orchestrator=verifier_orch,
        wm_manager=wm_manager,
        episodic_memory=episodic,
        semantic_memory=semantic,
    )

    return orch, episodic


def _load_dynamic_rules_from_db(db_path: str) -> DynamicRuleStore:
    """从 .db 的 dynamic_rules 表加载动态规则"""
    import json
    import sqlite3

    store = DynamicRuleStore()
    conn = sqlite3.connect(db_path, check_same_thread=False)
    try:
        rows = conn.execute(
            "SELECT rule_id, trigger_pattern, violation_message, "
            "trigger_count, false_positive_count FROM dynamic_rules"
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []
    finally:
        conn.close()

    for rid, pattern, msg, tc, fp in rows:
        store.add_rule({
            "rule_id": rid,
            "trigger_pattern": json.loads(pattern) if pattern else {},
            "violation_message": msg,
            "trigger_count": tc,
            "false_positive_count": fp,
        })
    return store
