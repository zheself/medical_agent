"""
memory/working_memory.py — 工作记忆（会话级）

存储当前会话的结构化状态：
- 患者档案 (patient_profile): 由 Planner 显式维护的症状档案
- 对话历史 (sliding window)
- 已检索证据池 (去重)
- 已探索的诊断 (避免循环)
- 压缩摘要 (超长时启用)

压缩策略：
1. raw_history 用 sliding window，保留最近 N 轮
2. 超出 token 上限时，老对话用小模型摘要
3. 结构化字段（patient_profile）永不压缩

实现状态: ✅ 完整
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from ..schemas import MemoryUpdate, PatientProfile


# ============================================================
# Token 计数器（用于压缩判断）
# ============================================================

def naive_token_count(text: str) -> int:
    """
    极简 token 估算：1 个汉字 ≈ 1.5 token，1 个英文单词 ≈ 1.3 token
    生产环境请用 transformers.AutoTokenizer
    """
    import re
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    english_words = len(re.findall(r"\b[a-zA-Z]+\b", text))
    return int(chinese_chars * 1.5 + english_words * 1.3)


# ============================================================
# Summarizer 抽象（用于压缩）
# ============================================================

class Summarizer:
    def summarize(self, texts: List[str], focus_on: Optional[List[str]] = None) -> str:
        raise NotImplementedError


class MockSummarizer(Summarizer):
    """Mock: 取前 N 字 + 提示。生产环境替换为 1.5B 小模型"""
    
    def summarize(self, texts: List[str], focus_on: Optional[List[str]] = None) -> str:
        full = " ".join(texts)
        focus_str = f"(关注: {', '.join(focus_on)})" if focus_on else ""
        return f"[历史摘要 {focus_str}]: {full[:200]}..."


# ============================================================
# Working Memory
# ============================================================

@dataclass
class WorkingMemory:
    """单次会话的工作记忆容器"""
    
    session_id: str = ""
    user_id: str = ""
    
    # 1. 原始对话（sliding window）
    raw_history: deque = field(default_factory=lambda: deque(maxlen=10))
    
    # 2. 结构化患者档案（核心，永不压缩）
    patient_profile: PatientProfile = field(default_factory=PatientProfile)
    
    # 3. 已检索证据池（去重）
    retrieved_evidence: Dict[str, List[Dict]] = field(default_factory=lambda: {
        "kg_facts": [],
        "literature": [],
        "drug_info": [],
        "community_summaries": [],
    })
    
    # 4. 已探索的诊断（避免重复推荐）
    explored_diagnoses: set = field(default_factory=set)
    
    # 5. 中间结论
    intermediate_conclusions: List[str] = field(default_factory=list)
    
    # 6. 压缩历史摘要（超长时启用）
    compressed_summary: str = ""
    
    # 配置
    max_tokens: int = 4000
    keep_recent_turns: int = 3


class WorkingMemoryManager:
    """Working Memory 的管理器，负责更新、压缩、序列化"""
    
    def __init__(self, summarizer: Optional[Summarizer] = None):
        self.summarizer = summarizer or MockSummarizer()
    
    def create(self, session_id: str, user_id: str) -> WorkingMemory:
        return WorkingMemory(session_id=session_id, user_id=user_id)
    
    def add_user_turn(self, wm: WorkingMemory, text: str) -> None:
        wm.raw_history.append({"role": "user", "text": text})
        self._maybe_compress(wm)
    
    def add_assistant_turn(self, wm: WorkingMemory, text: str) -> None:
        wm.raw_history.append({"role": "assistant", "text": text})
        self._maybe_compress(wm)
    
    def apply_planner_update(self, wm: WorkingMemory, update: MemoryUpdate) -> None:
        """应用 Planner 的 memory_update 指令"""
        if not update:
            return
        
        # 添加新症状（去重）
        for symptom in update.add_symptoms:
            existing_names = {s.get("name") if isinstance(s, dict) else s
                              for s in wm.patient_profile.symptoms}
            if symptom not in existing_names:
                wm.patient_profile.symptoms.append({"name": symptom})
        
        # 设置字段
        for k, v in update.set_fields.items():
            if hasattr(wm.patient_profile, k):
                setattr(wm.patient_profile, k, v)
        
        # 加入已探索诊断
        for diag in update.add_explored_diagnoses:
            wm.explored_diagnoses.add(diag)
        
        # 加入证据
        for evidence in update.add_evidence:
            ev_type = evidence.get("type", "kg_facts")
            if ev_type in wm.retrieved_evidence:
                wm.retrieved_evidence[ev_type].append(evidence)
    
    def add_tool_evidence(self, wm: WorkingMemory, tool_name: str, result: Dict) -> None:
        """工具执行后，把结果加入证据池"""
        if "facts" in result:
            wm.retrieved_evidence["kg_facts"].extend(result["facts"][:10])
        if "communities" in result:
            wm.retrieved_evidence["community_summaries"].extend(result["communities"])
    
    def get_planner_context(self, wm: WorkingMemory) -> Dict[str, Any]:
        """
        为 Planner 准备的上下文（卡片化）
        
        这是关键：用结构化卡片而非原始对话，token 消耗降 90%+
        """
        return {
            "patient_profile": asdict(wm.patient_profile),
            "explored_diagnoses": list(wm.explored_diagnoses),
            "recent_history": list(wm.raw_history)[-self._keep_recent(wm):],
            "compressed_summary": wm.compressed_summary[:500] if wm.compressed_summary else "",
            "evidence_count": {k: len(v) for k, v in wm.retrieved_evidence.items()},
        }
    
    def get_tokens(self, wm: WorkingMemory) -> int:
        """估算当前 working memory 占用的 token 数"""
        text = json.dumps(self.get_planner_context(wm), ensure_ascii=False)
        return naive_token_count(text)
    
    def _maybe_compress(self, wm: WorkingMemory) -> None:
        """超出阈值时触发压缩"""
        if self.get_tokens(wm) <= wm.max_tokens:
            return
        
        keep = self._keep_recent(wm)
        if len(wm.raw_history) <= keep:
            return  # 已经很短了，没法再压
        
        # 提取要压缩的部分
        old_turns = list(wm.raw_history)[:-keep]
        recent_turns = list(wm.raw_history)[-keep:]
        
        # 用 summarizer 压缩
        old_texts = [f"{t['role']}: {t['text']}" for t in old_turns]
        summary = self.summarizer.summarize(
            old_texts,
            focus_on=["symptoms", "tried_diagnoses", "user_concerns"]
        )
        
        # 累积到 compressed_summary
        wm.compressed_summary = (wm.compressed_summary + "\n" + summary).strip()
        wm.raw_history = deque(recent_turns, maxlen=wm.raw_history.maxlen)
    
    def _keep_recent(self, wm: WorkingMemory) -> int:
        return min(wm.keep_recent_turns, len(wm.raw_history))


# ============================================================
# 便捷的"患者卡片"格式化器（注入 prompt 时用）
# ============================================================

def format_patient_card(profile: PatientProfile) -> str:
    """
    把结构化档案转为紧凑的文本卡片
    示例输出（约 80 tokens vs 原始 1000+ tokens 对话）:
    
    [患者档案]
    - 年龄: 35 / 性别: 男
    - 主诉: 持续头痛伴发烧
    - 症状: 头痛(3天), 发烧(38.5°C), 颈强直
    - 过敏: 青霉素
    - 当前用药: 无
    [/患者档案]
    """
    lines = ["[患者档案]"]
    if profile.age or profile.gender:
        lines.append(f"- 年龄: {profile.age or '?'} / 性别: {profile.gender or '?'}")
    if profile.chief_complaint:
        lines.append(f"- 主诉: {profile.chief_complaint}")
    if profile.symptoms:
        sym_strs = [
            s["name"] + (f"({s.get('duration', '')})" if isinstance(s, dict) and s.get('duration') else "")
            for s in profile.symptoms if isinstance(s, dict) and "name" in s
        ]
        if sym_strs:
            lines.append(f"- 症状: {', '.join(sym_strs)}")
    if profile.allergies:
        lines.append(f"- 过敏: {', '.join(profile.allergies)}")
    if profile.medical_history:
        lines.append(f"- 病史: {', '.join(profile.medical_history)}")
    if profile.current_medications:
        lines.append(f"- 当前用药: {', '.join(profile.current_medications)}")
    lines.append("[/患者档案]")
    return "\n".join(lines)
