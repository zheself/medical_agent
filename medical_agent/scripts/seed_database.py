"""
scripts/seed_database.py — 生成模拟的真实数据库 data/medical_agent.db

为什么需要它:
- 之前 KG 是硬编码 Python dict（MockKGBackend），Episodic 用 :memory:。
- 本脚本生成一个真实的 SQLite .db 文件，让系统"连的是真库"，
  可被 SQLiteKGBackend / EpisodicMemory / SemanticMemory 共同读取。

⚠️ 数据是**编造但医学上合理**的小规模种子数据，仅用于演示和测试，
   不构成医疗知识库。生产环境请用 Huatuo-26M 构建的真实 Neo4j（见 build_kg.py）。

生成的表:
  - kg_entities      实体（疾病/症状/药物/...）
  - kg_relations     关系（三元组 + 权重）
  - kg_pagerank      预计算的 PageRank 分数
  - communities      GraphRAG 社区摘要
  - episodes         用户历史问诊（Episodic Memory）
  - failure_cases    L3 反思失败案例（Semantic Memory）
  - dynamic_rules    动态规则（Semantic Memory）

用法:
    python scripts/seed_database.py                 # 生成 data/medical_agent.db
    python scripts/seed_database.py --db ./x.db     # 指定路径
    python scripts/seed_database.py --force         # 覆盖已存在的库
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ============================================================
# 种子数据：知识图谱
# ============================================================

# 实体: (name, type)
ENTITIES = [
    # 疾病
    ("脑膜炎", "disease"), ("脑炎", "disease"), ("蛛网膜下腔出血", "disease"),
    ("偏头痛", "disease"), ("流感", "disease"), ("肺炎", "disease"),
    ("普通感冒", "disease"), ("糖尿病", "disease"), ("糖尿病视网膜病变", "disease"),
    ("糖尿病肾病", "disease"), ("高血压", "disease"), ("冠心病", "disease"),
    ("胃炎", "disease"), ("哮喘", "disease"),
    # 症状
    ("头痛", "symptom"), ("发烧", "symptom"), ("颈强直", "symptom"),
    ("畏光", "symptom"), ("咳嗽", "symptom"), ("咳痰", "symptom"),
    ("视力模糊", "symptom"), ("多饮", "symptom"), ("多尿", "symptom"),
    ("胸痛", "symptom"), ("乏力", "symptom"), ("肌肉酸痛", "symptom"),
    ("胃肠道不适", "symptom"),
    # 药物
    ("布洛芬", "drug"), ("对乙酰氨基酚", "drug"), ("青霉素", "drug"),
    ("头孢菌素", "drug"), ("二甲双胍", "drug"), ("胰岛素", "drug"),
    ("硝苯地平", "drug"), ("奥司他韦", "drug"), ("抗生素", "drug"),
    # 药物类别 / 治疗 / 检查
    ("双胍类降糖药", "drug_class"), ("降糖激素", "drug_class"),
    ("非甾体抗炎药", "drug_class"),
    ("腰椎穿刺", "examination"), ("眼底检查", "examination"),
    ("血糖检测", "examination"),
    ("神经内科", "department"), ("内分泌科", "department"),
    ("呼吸内科", "department"),
]

# 关系: (src, rel_type, dst, weight)
RELATIONS = [
    # 脑膜炎
    ("脑膜炎", "典型症状", "头痛", 0.90),
    ("脑膜炎", "典型症状", "发烧", 0.90),
    ("脑膜炎", "典型症状", "颈强直", 0.95),
    ("脑膜炎", "典型症状", "畏光", 0.70),
    ("脑膜炎", "推荐检查", "腰椎穿刺", 0.95),
    ("脑膜炎", "推荐治疗", "抗生素", 0.90),
    ("脑膜炎", "所属科室", "神经内科", 0.85),
    # 症状 -> 可能病因
    ("头痛", "可能病因", "偏头痛", 0.60),
    ("头痛", "可能病因", "脑膜炎", 0.30),
    ("头痛", "可能病因", "蛛网膜下腔出血", 0.10),
    ("头痛", "缓解药物", "布洛芬", 0.70),
    ("发烧", "可能病因", "流感", 0.50),
    ("发烧", "可能病因", "肺炎", 0.30),
    ("发烧", "可能病因", "脑膜炎", 0.20),
    ("发烧", "缓解药物", "对乙酰氨基酚", 0.90),
    ("颈强直", "可能病因", "脑膜炎", 0.85),
    ("颈强直", "可能病因", "蛛网膜下腔出血", 0.10),
    # 糖尿病
    ("糖尿病", "并发症", "糖尿病视网膜病变", 0.70),
    ("糖尿病", "并发症", "糖尿病肾病", 0.60),
    ("糖尿病", "典型症状", "多饮", 0.75),
    ("糖尿病", "典型症状", "多尿", 0.75),
    ("糖尿病", "推荐药物", "二甲双胍", 0.90),
    ("糖尿病", "推荐药物", "胰岛素", 0.80),
    ("糖尿病", "推荐检查", "血糖检测", 0.90),
    ("糖尿病", "所属科室", "内分泌科", 0.90),
    ("糖尿病视网膜病变", "典型症状", "视力模糊", 0.80),
    ("糖尿病视网膜病变", "推荐检查", "眼底检查", 0.90),
    ("视力模糊", "可能病因", "糖尿病视网膜病变", 0.55),
    # 流感 / 呼吸道
    ("流感", "典型症状", "发烧", 0.85),
    ("流感", "典型症状", "肌肉酸痛", 0.70),
    ("流感", "典型症状", "乏力", 0.65),
    ("流感", "推荐药物", "奥司他韦", 0.80),
    ("流感", "所属科室", "呼吸内科", 0.80),
    ("肺炎", "典型症状", "咳嗽", 0.80),
    ("肺炎", "典型症状", "咳痰", 0.75),
    ("肺炎", "典型症状", "发烧", 0.70),
    ("普通感冒", "典型症状", "咳嗽", 0.60),
    ("普通感冒", "缓解药物", "对乙酰氨基酚", 0.50),
    # 高血压 / 心血管
    ("高血压", "推荐药物", "硝苯地平", 0.80),
    ("高血压", "并发症", "冠心病", 0.50),
    ("冠心病", "典型症状", "胸痛", 0.70),
    # 药物 -> 类别 / 适应症 / 副作用（支持"X 是什么药"）
    ("二甲双胍", "药物类别", "双胍类降糖药", 0.95),
    ("二甲双胍", "适应症", "糖尿病", 0.95),
    ("二甲双胍", "常见副作用", "胃肠道不适", 0.60),
    ("胰岛素", "药物类别", "降糖激素", 0.95),
    ("胰岛素", "适应症", "糖尿病", 0.95),
    ("布洛芬", "药物类别", "非甾体抗炎药", 0.95),
    ("布洛芬", "适应症", "头痛", 0.80),
    ("布洛芬", "适应症", "发烧", 0.80),
    ("对乙酰氨基酚", "药物类别", "非甾体抗炎药", 0.85),
    ("青霉素", "药物类别", "抗生素", 0.90),
    ("头孢菌素", "药物类别", "抗生素", 0.90),
]

# GraphRAG 社区摘要: (community_id, level, theme, core_entities, summary)
COMMUNITIES = [
    ("L1_C001", 1, "中枢神经系统感染",
     ["脑膜炎", "脑炎", "头痛", "发烧", "颈强直", "蛛网膜下腔出血"],
     "本社区涵盖脑膜炎、脑炎等中枢神经系统感染性疾病。头痛+发热+颈强直三联征是脑膜炎的典型表现；"
     "蛛网膜下腔出血也可表现为剧烈头痛但通常为突发性。诊断金标准是腰椎穿刺脑脊液检查，治疗原则是早期经验性抗生素。"),
    ("L1_C002", 1, "糖尿病及其并发症",
     ["糖尿病", "糖尿病视网膜病变", "糖尿病肾病", "二甲双胍", "胰岛素", "视力模糊"],
     "糖尿病的慢性并发症主要分为微血管病变（视网膜病变、肾病）和大血管病变。视力模糊是糖尿病视网膜病变的早期信号。"
     "二甲双胍是 2 型糖尿病一线用药，胰岛素用于 1 型及晚期 2 型。"),
    ("L1_C003", 1, "上呼吸道感染与流感",
     ["流感", "普通感冒", "肺炎", "发烧", "咳嗽", "奥司他韦"],
     "本社区包括普通感冒、流感、肺炎等。流感典型表现为高热+全身肌肉酸痛+乏力；普通感冒多以鼻塞流涕为主。"
     "流感时可使用奥司他韦。"),
]

# Episodic: 模拟用户历史 (user_id, days_ago, type, diagnoses, medications, symptoms, summary)
EPISODES = [
    ("patient_001", 300, "diagnosis", ["糖尿病"], ["二甲双胍"], ["多饮", "多尿"],
     "确诊2型糖尿病，开始服用二甲双胍，提示青霉素过敏"),
    ("patient_001", 120, "consultation", ["偏头痛"], ["布洛芬"], ["头痛"],
     "偶发偏头痛，建议布洛芬缓解"),
    ("patient_001", 30, "consultation", ["糖尿病视网膜病变"], [], ["视力模糊"],
     "糖尿病随访发现视力模糊，建议眼底检查"),
    ("patient_002", 200, "diagnosis", ["高血压"], ["硝苯地平"], ["头晕"],
     "确诊高血压，服用硝苯地平控制"),
    ("patient_002", 15, "consultation", ["普通感冒"], ["对乙酰氨基酚"], ["咳嗽", "发烧"],
     "普通感冒伴低热，对症治疗"),
    ("patient_003", 60, "diagnosis", ["哮喘"], [], ["咳嗽"],
     "确诊支气管哮喘，对花粉过敏"),
]

# Semantic: 失败案例 (query, wrong_answer, errors, correction, root_cause_type, root_cause_detail)
FAILURE_CASES = [
    ("糖尿病人头痛能吃布洛芬吗",
     "可以服用布洛芬缓解头痛",
     ["未考虑糖尿病人长期用 NSAIDs 的肾损伤风险"],
     "糖尿病人长期服用布洛芬等 NSAIDs 会增加肾损伤风险，建议在医生指导下短期使用，优先对乙酰氨基酚",
     "missing_constraint",
     {"constraint": "糖尿病 + NSAIDs 需警示肾损伤", "message": "糖尿病人用 NSAIDs 应提示肾损伤风险"}),
]

# Semantic: 动态规则 (rule_id, trigger_pattern, violation_message, trigger_count, fp_count)
DYNAMIC_RULES = [
    ("auto_diabetes_nsaid",
     {"condition": "has_chronic:糖尿病", "drug_class": "非甾体抗炎药"},
     "糖尿病患者使用非甾体抗炎药需警示肾损伤风险",
     3, 0),
]


# ============================================================
# 建表
# ============================================================

SCHEMA = """
CREATE TABLE IF NOT EXISTS kg_entities (
    name TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    properties TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_entity_type ON kg_entities(type);

CREATE TABLE IF NOT EXISTS kg_relations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    src TEXT NOT NULL,
    rel_type TEXT NOT NULL,
    dst TEXT NOT NULL,
    weight REAL DEFAULT 1.0,
    confidence REAL DEFAULT 1.0,
    UNIQUE(src, rel_type, dst)
);
CREATE INDEX IF NOT EXISTS idx_rel_src ON kg_relations(src);
CREATE INDEX IF NOT EXISTS idx_rel_dst ON kg_relations(dst);

CREATE TABLE IF NOT EXISTS kg_pagerank (
    name TEXT PRIMARY KEY,
    score REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS communities (
    community_id TEXT PRIMARY KEY,
    level INTEGER NOT NULL,
    theme TEXT,
    core_entities TEXT,
    summary TEXT,
    match_keywords TEXT
);

CREATE TABLE IF NOT EXISTS episodes (
    episode_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    timestamp TEXT,
    episode_type TEXT,
    diagnoses TEXT,
    medications TEXT,
    symptoms TEXT,
    summary TEXT,
    importance_score REAL,
    access_count INTEGER DEFAULT 0,
    last_accessed TEXT,
    embedding TEXT
);
CREATE INDEX IF NOT EXISTS idx_user_time ON episodes(user_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_importance ON episodes(user_id, importance_score);

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

CREATE TABLE IF NOT EXISTS dynamic_rules (
    rule_id TEXT PRIMARY KEY,
    trigger_pattern TEXT,
    violation_message TEXT,
    trigger_count INTEGER DEFAULT 1,
    false_positive_count INTEGER DEFAULT 0,
    needs_verification BOOLEAN DEFAULT 1,
    source TEXT
);
"""


# ============================================================
# 计算 PageRank（用项目已有的纯 Python PPR）
# ============================================================

def compute_pagerank(relations):
    """用无 personalization 的 PageRank 给所有节点打分"""
    from src.tools.ppr_reasoner import SimpleGraph, personalized_pagerank
    g = SimpleGraph()
    for src, _, dst, w in relations:
        g.add_edge(src, dst, w)
    # 全节点作为 seed = 普通 PageRank
    all_nodes = list(g.nodes)
    pr = personalized_pagerank(g, all_nodes, alpha=0.85)
    return pr


def embed_text(text: str):
    """用项目的语义 Mock Embedder 生成向量"""
    from src.memory.mock_embedder import SemanticMockEmbedder
    return SemanticMockEmbedder().embed(text)


# ============================================================
# 主流程
# ============================================================

def seed(db_path: str, force: bool = False, quiet: bool = False):
    def log(*a):
        if not quiet:
            print(*a)
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not force:
            log(f"❌ {db_path} 已存在。用 --force 覆盖。")
            return False
        path.unlink()

    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)

    # 1. 实体
    conn.executemany(
        "INSERT OR REPLACE INTO kg_entities(name, type) VALUES (?, ?)",
        ENTITIES,
    )
    log(f"  实体: {len(ENTITIES)}")

    # 2. 关系
    conn.executemany(
        "INSERT OR REPLACE INTO kg_relations(src, rel_type, dst, weight) VALUES (?, ?, ?, ?)",
        RELATIONS,
    )
    log(f"  关系: {len(RELATIONS)}")

    # 3. PageRank
    pr = compute_pagerank(RELATIONS)
    conn.executemany(
        "INSERT OR REPLACE INTO kg_pagerank(name, score) VALUES (?, ?)",
        [(n, s) for n, s in pr.items()],
    )
    log(f"  PageRank: {len(pr)} 个节点")

    # 4. 社区（含 match_keywords，从 core_entities 取症状词便于关键词检索）
    for cid, level, theme, core, summary in COMMUNITIES:
        match_kw = [e for e in core if e in {"头痛", "发烧", "颈强直", "畏光", "视力模糊", "咳嗽", "多饮"}]
        conn.execute(
            "INSERT OR REPLACE INTO communities(community_id, level, theme, core_entities, summary, match_keywords) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (cid, level, theme, json.dumps(core, ensure_ascii=False),
             summary, json.dumps(match_kw, ensure_ascii=False)),
        )
    log(f"  社区: {len(COMMUNITIES)}")

    # 5. Episodes（计算 importance + embedding）
    from src.memory.episodic_memory import Episode, ImportanceScorer
    scorer = ImportanceScorer()
    now = datetime.now()
    n_ep = 0
    for user_id, days_ago, etype, diag, meds, symp, summary in EPISODES:
        ep = Episode(
            user_id=user_id, episode_type=etype,
            diagnoses=diag, medications=meds, symptoms=symp, summary=summary,
            timestamp=now - timedelta(days=days_ago),
        )
        ep.importance_score = scorer.score(ep)
        emb = embed_text(summary)
        conn.execute(
            "INSERT INTO episodes(episode_id, user_id, timestamp, episode_type, diagnoses, "
            "medications, symptoms, summary, importance_score, access_count, embedding) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (ep.episode_id, user_id, ep.timestamp.isoformat(), etype,
             json.dumps(diag, ensure_ascii=False), json.dumps(meds, ensure_ascii=False),
             json.dumps(symp, ensure_ascii=False), summary, ep.importance_score,
             0, json.dumps(emb)),
        )
        n_ep += 1
    log(f"  Episodes: {n_ep}")

    # 6. 失败案例
    for q, wrong, errs, corr, rc_type, rc_detail in FAILURE_CASES:
        conn.execute(
            "INSERT INTO failure_cases(query, wrong_answer, errors, correction, "
            "root_cause_type, root_cause_detail, embedding, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (q, wrong, json.dumps(errs, ensure_ascii=False), corr, rc_type,
             json.dumps(rc_detail, ensure_ascii=False),
             json.dumps(embed_text(q)), now.isoformat()),
        )
    log(f"  失败案例: {len(FAILURE_CASES)}")

    # 7. 动态规则
    for rid, pattern, msg, tc, fp in DYNAMIC_RULES:
        conn.execute(
            "INSERT OR REPLACE INTO dynamic_rules(rule_id, trigger_pattern, violation_message, "
            "trigger_count, false_positive_count, source) VALUES (?, ?, ?, ?, ?, ?)",
            (rid, json.dumps(pattern, ensure_ascii=False), msg, tc, fp, "seed"),
        )
    log(f"  动态规则: {len(DYNAMIC_RULES)}")

    conn.commit()
    conn.close()
    log(f"\n✅ 数据库已生成: {db_path}")
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=str(ROOT / "data" / "medical_agent.db"))
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    print(f"生成数据库: {args.db}\n")
    seed(args.db, force=args.force)


if __name__ == "__main__":
    main()
