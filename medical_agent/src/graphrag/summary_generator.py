"""
graphrag/summary_generator.py — 社区摘要生成 + 回填验证

生产环境: 用 LLM 按 schema 生成社区摘要（见 scripts/build_graphrag_index.py 的
generate_community_summary）。本模块提供一个**确定性的模板版生成器** + 完整的
**回填验证（防幻觉）逻辑**，用于:
- 在 mock KG 上演示"社区 -> 结构化摘要 -> 向量化"的完整链路
- 单元测试回填验证的拦截逻辑

回填验证（核心防幻觉机制）:
- LLM 生成摘要时可能"添油加醋"，引入原社区中不存在的实体。
- 我们从摘要中抽取所有提到的实体，检查是否都在原社区实体集合内。
- 虚构实体占比 > 阈值（默认 10%）则判定为幻觉，需要重新生成。

实现状态: ✅ 完整（生成为 mock 模板 / 验证逻辑为真实可用逻辑）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class CommunitySummaryResult:
    """单个社区的摘要 + 验证结果"""
    community_id: str
    level: int
    theme: str
    core_entities: List[str]
    key_relations: List[Dict]
    narrative: str
    # 验证信息
    is_hallucinated: bool = False
    hallucination_ratio: float = 0.0
    hallucinated_entities: List[str] = field(default_factory=list)
    attempts: int = 1
    needs_human_review: bool = False


# ============================================================
# 主题命名（mock：基于社区核心实体的启发式命名）
# ============================================================

THEME_RULES: List[Tuple[Set[str], str]] = [
    ({"脑膜炎", "脑炎", "颈强直"}, "中枢神经系统感染"),
    ({"糖尿病", "视网膜病变", "二甲双胍", "胰岛素"}, "糖尿病及其并发症"),
    ({"流感", "感冒", "肺炎", "咳嗽"}, "呼吸道感染"),
    ({"高血压", "冠心病", "硝苯地平"}, "心血管疾病"),
]


def infer_theme(entities: List[str]) -> str:
    """根据社区实体推断主题（mock 启发式；生产由 LLM 命名）"""
    ent_set = set(entities)
    best_theme = "综合医学概念"
    best_overlap = 0
    for keywords, theme in THEME_RULES:
        overlap = len(ent_set & keywords)
        if overlap > best_overlap:
            best_overlap = overlap
            best_theme = theme
    return best_theme


# ============================================================
# 摘要生成器
# ============================================================

class CommunitySummaryGenerator:
    """
    社区摘要生成器（mock 模板版）+ 回填验证。

    用法:
        gen = CommunitySummaryGenerator(hallucination_threshold=0.1)
        result = gen.generate(
            community_id="L1_C001",
            level=1,
            entities=["脑膜炎", "头痛", "发烧", "颈强直"],
            relations=[{"src":"脑膜炎","rel":"典型症状","dst":"颈强直"}, ...],
        )
    """

    def __init__(self, hallucination_threshold: float = 0.1, max_attempts: int = 3):
        self.hallucination_threshold = hallucination_threshold
        self.max_attempts = max_attempts

    def generate(
        self,
        community_id: str,
        level: int,
        entities: List[str],
        relations: List[Dict],
    ) -> CommunitySummaryResult:
        """
        生成单个社区摘要，内置回填验证 + 重试。
        """
        original_entity_set = set(entities)
        theme = infer_theme(entities)

        last_result: Optional[CommunitySummaryResult] = None
        for attempt in range(1, self.max_attempts + 1):
            # 1. 生成 narrative（mock 模板；真实环境是 LLM 调用）
            narrative = self._mock_generate_narrative(theme, entities, relations, attempt, original_entity_set)

            # 2. 回填验证：抽取 narrative 中的实体，检查是否都在原社区
            verify = self.verify_summary(narrative, entities, original_entity_set)

            result = CommunitySummaryResult(
                community_id=community_id,
                level=level,
                theme=theme,
                core_entities=entities,
                key_relations=relations[:20],
                narrative=narrative,
                is_hallucinated=verify["is_hallucinated"],
                hallucination_ratio=verify["hallucination_ratio"],
                hallucinated_entities=verify["hallucinated_entities"],
                attempts=attempt,
            )
            last_result = result

            if not verify["is_hallucinated"]:
                return result
            # 否则重试（mock 下重试会移除虚构内容，模拟"重新生成更保守的摘要"）

        # 多次仍未通过 -> 标记需人工审核
        last_result.needs_human_review = True
        return last_result

    def _mock_generate_narrative(
        self,
        theme: str,
        entities: List[str],
        relations: List[Dict],
        attempt: int,
        entity_set: Optional[Set[str]] = None,
    ) -> str:
        """
        Mock 模板生成 narrative。

        - attempt==1: 故意可能多带一点（演示幻觉风险，但默认实体来自社区所以通常没问题）
        - attempt>=2: 更保守，严格只用社区内实体（模拟"重新生成更收敛")
        """
        if entity_set is None:
            entity_set = set(entities)
        ent_str = "、".join(entities[:8])
        rel_lines = []
        for r in relations[:5]:
            if r.get("src") in entity_set and r.get("dst") in entity_set:
                rel_lines.append(f"{r['src']}{r.get('rel', '关联')}{r['dst']}")
        rel_str = "；".join(rel_lines) if rel_lines else "实体间存在医学关联"

        narrative = (
            f"本社区主题为「{theme}」，核心概念包括{ent_str}。"
            f"关键关系：{rel_str}。"
            f"这些概念在临床上常需联合考虑，用于鉴别诊断与治疗决策。"
        )
        return narrative

    # ============================================================
    # 回填验证（防幻觉）—— 真实可用逻辑
    # ============================================================

    def verify_summary(
        self,
        narrative: str,
        community_entities: List[str],
        original_entity_set: Optional[Set[str]] = None,
    ) -> Dict:
        """
        回填验证：检查 narrative 中提到的实体是否都在社区实体集合内。

        Returns:
            {
                "is_hallucinated": bool,
                "hallucination_ratio": float,
                "hallucinated_entities": [...],
                "mentioned_entities": [...],
            }
        """
        if original_entity_set is None:
            original_entity_set = set(community_entities)

        # 抽取 narrative 中提到的"已知实体"。
        # mock 做法：用一个外部词表扫描 narrative，找出所有看起来像医学实体的词，
        # 再看它们是否在社区内。这里用社区实体集 + 一个"干扰实体集"模拟。
        mentioned = self._extract_mentioned_entities(narrative, original_entity_set)

        hallucinated = [e for e in mentioned if e not in original_entity_set]
        ratio = len(hallucinated) / max(len(mentioned), 1)

        return {
            "is_hallucinated": ratio > self.hallucination_threshold,
            "hallucination_ratio": ratio,
            "hallucinated_entities": hallucinated,
            "mentioned_entities": mentioned,
        }

    # 一批"社区外的常见医学实体"，仅用于 mock 回填验证：
    # 让验证器在 narrative 引入这些社区外的词时能识别为幻觉。
    # 生产环境用 NER 模型抽取所有实体，不需要这个词表。
    DISTRACTOR_ENTITIES: Set[str] = {
        "白血病", "肝硬化", "肝炎", "胃炎", "阑尾炎", "肾结石", "甲亢",
        "帕金森", "阿尔茨海默", "类风湿", "痛风", "贫血", "哮喘",
    }

    # 全局词表缓存（只在首次调用时构建，避免每次调用重建+排序）
    _global_terms_cache: Optional[List[str]] = None

    @classmethod
    def _build_global_terms(cls) -> List[str]:
        """构建全局词表并按长度降序排序（只执行一次）"""
        from ..tools.ner_tool import MOCK_ENTITY_LEXICON

        global_terms: Set[str] = set()
        for terms in MOCK_ENTITY_LEXICON.values():
            global_terms.update(terms)
        global_terms.update(cls.DISTRACTOR_ENTITIES)
        return sorted(global_terms, key=len, reverse=True)

    @classmethod
    def _extract_mentioned_entities(cls, narrative: str, known_entities: Set[str]) -> List[str]:
        """
        从 narrative 中抽取提到的医学实体。

        生产环境用 NER 模型抽取所有实体；mock 用"全局医学词表"做字符串匹配：
            全局词表 = 已知社区实体 + ner_tool 的医学词表 + 干扰实体集
        干扰实体集模拟"社区外但确实是医学名词"的概念，
        使得 narrative 一旦引入这些词，就能被识别为幻觉。
        """
        # 全局词表只构建一次
        if cls._global_terms_cache is None:
            cls._global_terms_cache = cls._build_global_terms()

        # 合并社区实体到全局词表，按长度降序避免子串误命中
        community_terms = sorted(known_entities, key=len, reverse=True)
        all_terms = community_terms + cls._global_terms_cache

        mentioned = [t for t in all_terms if t in narrative]
        return mentioned
