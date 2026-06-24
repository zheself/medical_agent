"""
schemas.py — 全局共享的核心数据结构

设计原则：
1. 使用 dataclass + pydantic 风格，保证序列化和类型安全
2. 所有模块通过这些 schema 通信，避免散乱的 dict
3. Schema 即文档：看 schema 就能理解系统

实现状态: ✅ 完整
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from datetime import datetime
import uuid


# ============================================================
# KG 实体类型 & 关系类型（统一 Schema）
# ============================================================

ENTITY_TYPES = {
    "disease": ["疾病", "病症"],
    "symptom": ["症状"],
    "drug": ["药物", "药品"],
    "treatment": ["治疗", "疗法", "手术"],
    "examination": ["检查", "检验"],
    "department": ["科室"],
    "body_part": ["部位"],
    "food": ["食物"],
}

RELATION_TYPES = [
    # 疾病 → 其他实体
    "典型症状",    # disease → symptom
    "可能病因",    # symptom → disease（反向边）
    "推荐药物",    # disease → drug
    "禁用药物",    # disease → drug
    "并发症",      # disease → disease
    "推荐治疗",    # disease → treatment
    "推荐检查",    # disease → examination
    "所属科室",    # disease → department
    "影响部位",    # disease → body_part
    # 症状/药物相关
    "缓解药物",    # symptom → drug
    "适应症",      # drug → disease
    "副作用",      # drug → symptom
    "禁忌症",      # drug → disease
    # 食物相关
    "宜吃食物",    # disease → food
    "忌吃食物",    # disease → food
]


# ============================================================
# 用户输入与会话
# ============================================================

class ComplexityLevel(str, Enum):
    """查询复杂度，决定后续走多深的反思"""
    LOW = "low"        # 简单事实查询，仅 L1
    MEDIUM = "medium"  # 中等推理，L1 + L2
    HIGH = "high"      # 复杂诊断/鉴别，可能 L1+L2+L3


class QueryMode(str, Enum):
    """检索模式，由 Planner 或 Router 决策"""
    LOCAL = "local"    # 实体级检索
    GLOBAL = "global"  # 社区级检索
    HYBRID = "hybrid"  # 两个都跑


@dataclass
class UserQuery:
    """单次用户输入"""
    query_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    text: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    # 可选的附加上下文
    attachments: List[Dict[str, Any]] = field(default_factory=list)


# ============================================================
# Planner 输出
# ============================================================

@dataclass
class PlanStep:
    """计划中的单步任务"""
    step_id: int
    tool: str                                  # 调用的工具名
    input: Any                                  # 工具输入（可引用其他 step 的输出）
    depends_on: List[int] = field(default_factory=list)
    # 输出字段：执行后填充
    output: Optional[Any] = None
    status: str = "pending"                    # pending/running/done/failed
    error: Optional[str] = None
    elapsed_ms: Optional[float] = None


@dataclass
class MemoryUpdate:
    """Planner 显式更新 Working Memory 的指令"""
    add_symptoms: List[str] = field(default_factory=list)
    set_fields: Dict[str, Any] = field(default_factory=dict)
    add_explored_diagnoses: List[str] = field(default_factory=list)
    add_evidence: List[Dict] = field(default_factory=list)


@dataclass
class Plan:
    """Planner 输出的完整计划"""
    thought: str
    complexity: ComplexityLevel
    steps: List[PlanStep] = field(default_factory=list)
    # 推测式预取的候选（可选）
    speculative_prefetch: List[str] = field(default_factory=list)
    expected_output_type: str = "answer"
    memory_update: Optional[MemoryUpdate] = None
    diagnoses: List[str] = field(default_factory=list)  # LLM 的诊断结论列表


# ============================================================
# 工具执行结果
# ============================================================

@dataclass
class ToolResult:
    """统一的工具返回格式"""
    tool_name: str
    success: bool
    data: Any = None
    error: Optional[str] = None
    elapsed_ms: float = 0.0
    # 引用信息（用于后续 citation accuracy 评测）
    citations: List[Dict] = field(default_factory=list)


# ============================================================
# 知识图谱相关
# ============================================================

@dataclass
class KGEntity:
    """KG 实体"""
    id: str
    name: str
    type: str                                  # disease/symptom/drug/...
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KGRelation:
    """KG 关系（三元组）"""
    src: str
    rel_type: str
    dst: str
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0                    # 用于 Reflexion 沉淀的关系


@dataclass
class CommunitySummary:
    """GraphRAG 的社区摘要"""
    community_id: str
    level: int                                 # 0=粗 / 1=中 / 2=细
    theme: str
    core_entities: List[str]
    key_relations: List[Dict]
    narrative: str                             # 自然语言描述（向量化）
    embedding: Optional[List[float]] = None
    # 元数据
    size: int = 0
    confidence: float = 1.0


# ============================================================
# 反思与校验
# ============================================================

class VerifierLevel(str, Enum):
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"


@dataclass
class VerifyResult:
    """Verifier 的输出"""
    passed: bool
    level: VerifierLevel
    errors: List[str] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    # 是否需要升级到更高级 Verifier
    escalate: bool = False
    # 修正建议（若有）
    correction_hint: Optional[str] = None
    elapsed_ms: float = 0.0


@dataclass
class ReflexionRecord:
    """L3 Reflexion 的修正记录，写入 Semantic Memory"""
    query: str
    wrong_answer: str
    errors: List[str]
    correction: str
    root_cause_type: str                       # missing_relation/missing_constraint/other
    root_cause_detail: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


# ============================================================
# Memory 相关
# ============================================================

@dataclass
class PatientProfile:
    """Working Memory 中的结构化患者档案"""
    age: Optional[int] = None
    gender: Optional[str] = None
    chief_complaint: Optional[str] = None
    symptoms: List[Dict] = field(default_factory=list)         # [{"name": "头痛", "duration": "3天"}]
    duration: Optional[str] = None
    medical_history: List[str] = field(default_factory=list)
    current_medications: List[str] = field(default_factory=list)
    allergies: List[str] = field(default_factory=list)


@dataclass
class Episode:
    """Episodic Memory 中的单条事件"""
    episode_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    episode_type: str = "consultation"                          # consultation/diagnosis/medication/test_result
    diagnoses: List[str] = field(default_factory=list)
    medications: List[str] = field(default_factory=list)
    symptoms: List[str] = field(default_factory=list)
    summary: str = ""
    importance_score: float = 0.0
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    embedding: Optional[List[float]] = None


# ============================================================
# 最终回答
# ============================================================

@dataclass
class FinalAnswer:
    """Agent 的最终输出"""
    content: str                                                # 自然语言回答
    diagnoses: List[str] = field(default_factory=list)          # LLM 的诊断结论列表
    citations: List[Dict] = field(default_factory=list)         # 引用的 KG 事实和文献
    reasoning_trace: List[str] = field(default_factory=list)    # 推理轨迹（可解释性）
    confidence: float = 0.0
    # 元数据（用于 eval）
    plan: Optional[Plan] = None
    verification_results: List[VerifyResult] = field(default_factory=list)
    total_elapsed_ms: float = 0.0
    total_tokens: int = 0
    # 警告/免责声明
    warnings: List[str] = field(default_factory=list)


# ============================================================
# Eval 用的标准格式
# ============================================================

@dataclass
class EvalItem:
    """单条评测数据"""
    item_id: str
    query: str
    gold_answer: Optional[str] = None
    gold_diagnoses: List[str] = field(default_factory=list)
    gold_reasoning_path: List[str] = field(default_factory=list)
    min_hops_required: int = 1
    difficulty: str = "easy"                                    # easy/medium/hard
    # 多轮对话评测时使用
    history: List[Dict] = field(default_factory=list)
    user_id: Optional[str] = None
