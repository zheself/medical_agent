"""
mock_embedder.py — 确定性的"语义" Mock Embedder

为什么需要它:
- 原来的 MockEmbedder 用 md5(text) 生成伪向量，任意两段文本的相似度近似随机，
  完全无法演示"语义相近的文本相似度高"这个 RAG 的核心假设。
- 这个版本基于一个固定的医学概念词表，把文本映射成词表上的多热向量，
  并对同义/相关概念做轻量扩展。共享越多医学概念的文本，余弦相似度越高。

这样在没有真实 embedding 模型时，也能让 Episodic 混合检索、Global Search 等
检索逻辑展现出**可解释、可复现**的检索行为（例如老的"过敏史" episode
在问到相关症状时被正确召回）。

生产环境直接换成 BAAI/bge-m3 等真实模型即可，接口完全一致。

实现状态: ✅ 完整（确定性，无随机，无外部依赖）
"""
from __future__ import annotations

import math
from typing import Dict, List


# ============================================================
# 医学概念词表 (vocabulary)
# 每个 key 是一个"概念维度"，value 是触发该维度的关键词（含同义词）
# 文本命中任一关键词，对应维度就被激活
# ============================================================

MEDICAL_VOCAB: Dict[str, List[str]] = {
    # —— 症状维度 ——
    "dim_headache": ["头痛", "头疼", "偏头痛"],
    "dim_fever": ["发烧", "发热", "高热", "体温"],
    "dim_neck_stiff": ["颈强直", "颈部僵硬", "脖子硬"],
    "dim_cough": ["咳嗽", "咳痰", "干咳"],
    "dim_vision": ["视力模糊", "视物模糊", "视力下降", "视网膜"],
    "dim_thirst": ["多饮", "多尿", "口渴"],
    "dim_chest_pain": ["胸痛", "胸闷", "心悸"],
    "dim_abdomen": ["腹痛", "腹泻", "恶心", "呕吐"],
    # —— 疾病维度 ——
    "dim_meningitis": ["脑膜炎", "脑炎", "中枢神经感染"],
    "dim_diabetes": ["糖尿病", "血糖", "二型糖尿病", "2型糖尿病"],
    "dim_hypertension": ["高血压", "血压"],
    "dim_flu": ["流感", "感冒", "上呼吸道感染"],
    "dim_pneumonia": ["肺炎"],
    "dim_retinopathy": ["视网膜病变", "黄斑水肿", "眼底"],
    "dim_sah": ["蛛网膜下腔出血", "脑出血"],
    # —— 药物维度 ——
    "dim_metformin": ["二甲双胍", "降糖药"],
    "dim_insulin": ["胰岛素"],
    "dim_ibuprofen": ["布洛芬", "对乙酰氨基酚", "退烧药", "解热"],
    "dim_penicillin": ["青霉素", "头孢", "抗生素"],
    # —— 重要性 / 安全维度 ——
    "dim_allergy": ["过敏", "过敏史", "过敏反应"],
    "dim_emergency": ["急诊", "立即就医", "急救", "危重"],
    "dim_chronic": ["慢性", "长期", "终身"],
}

# 概念之间的弱关联（用于轻量扩展：命中 A 时，给 B 一个较小的激活）
# 让"症状"能召回"对应疾病"的 episode，提升检索的语义连贯性
CONCEPT_LINKS: Dict[str, List[str]] = {
    "dim_headache": ["dim_meningitis"],
    "dim_fever": ["dim_meningitis", "dim_flu", "dim_pneumonia"],
    "dim_neck_stiff": ["dim_meningitis"],
    "dim_vision": ["dim_diabetes", "dim_retinopathy"],
    "dim_thirst": ["dim_diabetes"],
    "dim_diabetes": ["dim_metformin", "dim_insulin", "dim_retinopathy"],
    "dim_meningitis": ["dim_emergency", "dim_penicillin"],
    "dim_penicillin": ["dim_allergy"],
}

# 固定维度顺序（保证向量可复现）
_DIM_ORDER: List[str] = sorted(MEDICAL_VOCAB.keys())
_DIM_INDEX: Dict[str, int] = {d: i for i, d in enumerate(_DIM_ORDER)}

# 激活权重
_PRIMARY_WEIGHT = 1.0   # 直接命中关键词
_LINKED_WEIGHT = 0.35   # 弱关联扩展


class SemanticMockEmbedder:
    """
    确定性语义 Mock Embedder。

    embed(text) 返回固定维度的向量：
    - 文本命中某概念关键词 -> 该维度 += 1.0
    - 通过 CONCEPT_LINKS 弱关联的维度 -> += 0.35

    两段共享医学概念越多的文本，余弦相似度越高。
    """

    dim: int = len(_DIM_ORDER)

    def embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        text = text or ""
        activated: List[str] = []

        # 1. 直接命中
        for dim_name, keywords in MEDICAL_VOCAB.items():
            if any(kw in text for kw in keywords):
                vec[_DIM_INDEX[dim_name]] += _PRIMARY_WEIGHT
                activated.append(dim_name)

        # 2. 弱关联扩展
        for dim_name in activated:
            for linked in CONCEPT_LINKS.get(dim_name, []):
                vec[_DIM_INDEX[linked]] += _LINKED_WEIGHT

        # 3. 全零兜底：避免无医学概念的文本导致 cosine 除零
        #    用一个稳定的微弱信号（基于文本长度）占位
        if not any(v > 0 for v in vec):
            if text:
                vec[len(text) % self.dim] = 0.01
        return vec

    @staticmethod
    def explain(text: str) -> List[str]:
        """调试用：返回文本激活了哪些概念维度"""
        return [
            dim_name for dim_name, keywords in MEDICAL_VOCAB.items()
            if any(kw in (text or "") for kw in keywords)
        ]
