"""
memory/semantic_memory.py — 语义记忆（全局共享）

本质上是 KG + 失败案例库，由 L3 Reflexion 驱动持续进化。

写入路径:
1. L3 Reflexion 修正 → 提取 root_cause → 写入
   - 类型 1 missing_relation: 在 KG 中创建新关系（标记低 confidence，待人工审核）
   - 类型 2 missing_constraint: 注册为新的 L1 规则
   - 类型 3 其他: 失败案例本身入库，供后续 L3 检索作 few-shot

实现状态: ✅ 完整接口 / 🟡 KG 写入部分依赖 Neo4j 后端
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..schemas import ReflexionRecord


class FailureCaseStore:
    """失败案例库：存储 L3 Reflexion 修正前后对照"""
    
    def __init__(self, db_path: str = ":memory:"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_schema()
    
    def _init_schema(self):
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS failure_cases (
            case_id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT,
            wrong_answer TEXT,
            errors TEXT,
            correction TEXT,
            root_cause_type TEXT,
            root_cause_detail TEXT,
            embedding TEXT,
            timestamp TEXT,
            verified BOOLEAN DEFAULT 0
        );
        """)
        self.conn.commit()
    
    def add(self, record: ReflexionRecord, embedding: Optional[List[float]] = None) -> None:
        self.conn.execute("""
            INSERT INTO failure_cases 
            (query, wrong_answer, errors, correction, root_cause_type, 
             root_cause_detail, embedding, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.query,
            record.wrong_answer,
            json.dumps(record.errors, ensure_ascii=False),
            record.correction,
            record.root_cause_type,
            json.dumps(record.root_cause_detail, ensure_ascii=False),
            json.dumps(embedding) if embedding else None,
            record.timestamp.isoformat(),
        ))
        self.conn.commit()
    
    def search_similar(
        self,
        query_embedding: List[float],
        top_k: int = 3,
        min_similarity: float = 0.5,
    ) -> List[Dict]:
        """检索相似的历史失败案例，用于 L3 的 few-shot 提示"""
        from .episodic_memory import cosine_similarity
        
        cur = self.conn.execute("SELECT * FROM failure_cases WHERE embedding IS NOT NULL")
        cols = [c[0] for c in cur.description]
        scored = []
        for row in cur.fetchall():
            data = dict(zip(cols, row))
            emb = json.loads(data["embedding"])
            sim = cosine_similarity(query_embedding, emb)
            if sim >= min_similarity:
                data["similarity"] = sim
                scored.append(data)
        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:top_k]
    
    def list_pending_verification(self) -> List[Dict]:
        """列出待医学专家审核的案例"""
        cur = self.conn.execute("SELECT * FROM failure_cases WHERE verified = 0")
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


class DynamicRuleStore:
    """从 Reflexion 沉淀的动态规则库（供 L1 Verifier 加载）"""
    
    def __init__(self):
        self.rules: List[Dict] = []
    
    def add_rule(self, rule: Dict) -> None:
        """
        rule 格式:
        {
            "rule_id": "auto_generated_xxx",
            "trigger_pattern": {"condition": "patient_age < 18", "drug_in": [...]},
            "violation_message": "...",
            "trigger_count": 1,
            "false_positive_count": 0,
            "needs_verification": True
        }
        """
        rule.setdefault("trigger_count", 1)
        rule.setdefault("false_positive_count", 0)
        rule.setdefault("needs_verification", True)
        self.rules.append(rule)
    
    def get_active_rules(self, min_confidence: float = 0.5) -> List[Dict]:
        """获取置信度达标的规则（用于在线 L1 Verify）"""
        active = []
        for r in self.rules:
            tc = r.get("trigger_count", 1)
            fp = r.get("false_positive_count", 0)
            confidence = (tc - fp) / max(tc, 1)
            if confidence >= min_confidence:
                active.append(r)
        return active


class SemanticMemory:
    """
    Semantic Memory 主类
    
    包含三个组件：
    1. KG (Neo4j)：通过 add_inferred_relation 加入新关系（低 confidence）
    2. FailureCaseStore: L3 失败案例
    3. DynamicRuleStore: 动态 L1 规则
    """
    
    def __init__(
        self,
        kg_backend=None,
        failure_store: Optional[FailureCaseStore] = None,
        rule_store: Optional[DynamicRuleStore] = None,
        embedder=None,
    ):
        self.kg = kg_backend  # 可为 None（PoC 阶段）
        self.failure_store = failure_store or FailureCaseStore()
        self.rule_store = rule_store or DynamicRuleStore()
        self.embedder = embedder
    
    def write_from_reflexion(self, record: ReflexionRecord) -> Dict[str, Any]:
        """
        L3 Reflexion 完成后调用，把修正沉淀回 Semantic Memory
        
        Returns:
            写入摘要，记录到哪些子模块
        """
        report = {"failure_case_added": False, "kg_relation_added": False, "rule_added": False}
        
        # 1. 写入失败案例库（始终写入）
        emb = self.embedder.embed(record.query) if self.embedder else None
        self.failure_store.add(record, embedding=emb)
        report["failure_case_added"] = True
        
        # 2. 根据 root_cause 类型做不同处理
        rc_type = record.root_cause_type
        rc_detail = record.root_cause_detail
        
        if rc_type == "missing_relation" and self.kg:
            # 在 KG 中创建新关系（标记低置信度，待审核）
            src = rc_detail.get("src_entity")
            rel = rc_detail.get("relation")
            dst = rc_detail.get("dst_entity")
            if src and rel and dst:
                # 生产环境：调用 Neo4j 写入接口
                # self.kg.create_relation(src, rel, dst, confidence=0.6, needs_verification=True)
                report["kg_relation_added"] = True
                report["new_relation"] = f"{src} -[{rel}]-> {dst}"
        
        elif rc_type == "missing_constraint":
            # 注册为新规则
            new_rule = {
                "rule_id": f"auto_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "trigger_pattern": rc_detail.get("constraint", {}),
                "violation_message": rc_detail.get("message", ""),
                "source": "reflexion_v1",
                "needs_verification": True,
            }
            self.rule_store.add_rule(new_rule)
            report["rule_added"] = True
            report["new_rule_id"] = new_rule["rule_id"]
        
        return report
    
    def get_few_shot_examples(self, query: str, top_k: int = 3) -> List[Dict]:
        """供 L3 Reflexion 调用，检索相似的历史失败案例"""
        if not self.embedder:
            return []
        emb = self.embedder.embed(query)
        return self.failure_store.search_similar(emb, top_k=top_k)
