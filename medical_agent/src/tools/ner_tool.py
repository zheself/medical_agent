"""
tools/ner_tool.py — 医疗命名实体识别工具

实现思路:
- 生产环境：BERT-BiLSTM-CRF（你简历里提到的）+ 医疗专用词表
- 当前实现：基于词表匹配的 Mock，足以让 demo 跑通

实现状态: 🟡 Mock 实现 / ⚪ 真实模型需要训练数据
"""
from __future__ import annotations

from typing import Any, Dict, List

from .base import BaseTool


# Mock 词表（生产环境应替换为完整医学术语库，如 CN-DBpedia + ICD-10）
MOCK_ENTITY_LEXICON = {
    "symptom": [
        "头痛", "发烧", "发热", "咳嗽", "咳痰", "胸痛", "腹痛", "腹泻",
        "恶心", "呕吐", "畏光", "颈强直", "颈部僵硬", "视力模糊",
        "气短", "心悸", "乏力", "嗜睡", "皮疹", "出血", "黄疸",
    ],
    "disease": [
        "糖尿病", "高血压", "冠心病", "脑膜炎", "流感", "肺炎",
        "胃炎", "肝炎", "偏头痛", "蛛网膜下腔出血", "脑炎",
        "视网膜病变", "黄斑水肿", "哮喘",
    ],
    "drug": [
        "阿司匹林", "布洛芬", "对乙酰氨基酚", "青霉素", "头孢", "甲硝唑",
        "二甲双胍", "胰岛素", "硝苯地平", "氨氯地平",
    ],
    "duration": [
        "三天", "一周", "一个月", "数年", "突发", "持续",
    ],
}


class NERTool(BaseTool):
    name = "ner"
    description = "从用户文本中抽取医疗实体（症状、疾病、药物、时长等）。返回结构化实体列表。"
    
    def __init__(self, lexicon: Dict[str, List[str]] = None):
        self.lexicon = lexicon or MOCK_ENTITY_LEXICON
    
    async def _execute(self, input_data: Any) -> Dict[str, List[str]]:
        # 兼容字符串和 dict 输入
        if isinstance(input_data, dict):
            text = input_data.get("text") or input_data.get("query") or ""
        else:
            text = str(input_data)
        
        result = {ent_type: [] for ent_type in self.lexicon}
        for ent_type, terms in self.lexicon.items():
            for term in terms:
                if term in text and term not in result[ent_type]:
                    result[ent_type].append(term)
        
        # 提供一个扁平化的列表方便下游使用
        result["all_entities"] = [
            e for ents in result.values() for e in ents
            if isinstance(ents, list)
        ]
        return result


# ============================================================
# 生产实现的接口骨架（供后续扩展）
# ============================================================

class BertBiLSTMCRFNERTool(BaseTool):
    """
    基于 BERT-BiLSTM-CRF 的 NER 工具（生产实现骨架）
    
    训练流程见 scripts/train_ner.py
    """
    name = "ner"
    description = "基于 BERT-BiLSTM-CRF 的医疗 NER，支持症状、疾病、药物等多类实体。"
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.model = None
        # self._load_model()  # 真实环境时启用
    
    def _load_model(self):
        """
        真实实现要点：
        - from transformers import BertTokenizer, BertModel
        - 自定义 BiLSTM + CRF 头
        - 加载 fine-tuned 权重
        """
        raise NotImplementedError("请在 GPU 环境中实现")
    
    async def _execute(self, input_data: Any) -> Dict[str, List[str]]:
        raise NotImplementedError
