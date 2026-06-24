"""
scripts/import_medical_kg.py — 从 medical_new_2.json 导入真实 KG 数据到 SQLite

执行顺序:
1. 读取 JSON, 抽取实体+关系
2. 过滤脏数据(纯英文短词 + 人名黑名单)
3. 批量写入 kg_entities / kg_relations 表(executemany + 事务)
4. 用 networkx 重算 PageRank
5. 用 igraph + leidenalg 重建三层社区检测
6. 对每个社区生成摘要(CommunitySummaryGenerator)

用法:
    python scripts/import_medical_kg.py --db data/medical_agent.db --sample 1000  # 小样本测耗时
    python scripts/import_medical_kg.py --db data/medical_agent.db               # 全量导入
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
import time
from pathlib import Path

# 添加项目根目录到 path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.schemas import ENTITY_TYPES, RELATION_TYPES


# ============================================================
# 脏数据过滤
# ============================================================

# 规则 1: 纯英文 1-3 字母 → 删除
PURE_ENGLISH_SHORT = re.compile(r'^[A-Za-z]{1,3}$')

# 规则 2: 人名黑名单 — 当前为空，低频人名在图里几乎是孤立节点，
# 对 PageRank/社区检测/PPR 影响微乎其微，暂不清洗。
# 等全量跑出来发现具体人名干扰了再回头删。
PERSON_NAME_BLACKLIST: set = set()


def is_dirty_entity(name: str) -> bool:
    """保守过滤: 只删纯英文短词和已知人名,绝不误删医学术语"""
    if PURE_ENGLISH_SHORT.match(name):
        return True
    if name in PERSON_NAME_BLACKLIST:
        return True
    return False


# ============================================================
# JSON 数据读取
# ============================================================

def load_records(path: str) -> list:
    """读取 medical_new_2.json(MongoDB export,每行一个 JSON 对象)"""
    records = []
    with open(path, encoding='utf-8') as f:
        content = f.read()
    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(content):
        try:
            obj, new_pos = decoder.raw_decode(content, pos)
            records.append(obj)
            pos = new_pos
            while pos < len(content) and content[pos] in ' \n\r\t':
                pos += 1
        except json.JSONDecodeError:
            pos += 1
            continue
    return records


# ============================================================
# 实体 & 关系抽取
# ============================================================

# 关系映射: (关系类型, 源类型, 目标类型, 权重)
RELATION_MAP = {
    "symptom":           ("典型症状", "disease", "symptom",    0.8),
    "common_drug":       ("推荐药物", "disease", "drug",       0.85),
    "recommand_drug":    ("推荐药物", "disease", "drug",       0.85),
    "check":             ("推荐检查", "disease", "examination", 0.8),
    "cure_department":   ("所属科室", "disease", "department",  0.9),
    "cure_way":          ("推荐治疗", "disease", "treatment",  0.8),
    "acompany":          ("并发症",   "disease", "disease",    0.6),
    "do_eat":            ("宜吃食物", "disease", "food",        0.5),
    "not_eat":           ("忌吃食物", "disease", "food",        0.5),
}

# 反向边: 症状→可能病因→疾病(权重 0.4)
REVERSE_REL = ("可能病因", "symptom", "disease", 0.4)

# 疾病属性字段(存到 kg_entities.properties JSON)
DISEASE_PROPS_FIELDS = ["desc", "cause", "prevent", "cost_money",
                        "cure_lasttime", "cured_prob", "get_prob",
                        "get_way", "yibao_status", "category"]


def extract_entities_and_relations(records, sample_size=None):
    """从 JSON 记录中抽取实体和关系"""
    if sample_size:
        records = records[:sample_size]

    entities = {}   # (name, type) → properties_json
    relations = []  # (src, rel_type, dst, weight)

    dirty_deleted = {}  # name → 删除原因
    dirty_rel_impact = 0  # 因脏数据删除而连带删除的关系数

    for r in records:
        disease_name = r.get("name", "").strip()
        if not disease_name:
            continue

        # 疾病实体 + 属性
        props = {}
        for field in DISEASE_PROPS_FIELDS:
            val = r.get(field)
            if val and (val if isinstance(val, str) else str(val).strip()):
                props[field] = val
        entities[(disease_name, "disease")] = json.dumps(props, ensure_ascii=False)

        # 抽取各类关系
        for json_field, (rel_type, src_type, dst_type, weight) in RELATION_MAP.items():
            raw_vals = r.get(json_field, [])
            if isinstance(raw_vals, str):
                raw_vals = [raw_vals] if raw_vals.strip() else []

            for val in raw_vals:
                val = val.strip()
                if not val:
                    continue

                # 脏数据过滤
                if is_dirty_entity(val):
                    dirty_deleted.setdefault(val, "黑名单/纯英文")
                    dirty_rel_impact += 1
                    continue

                # 实体注册
                entities[(val, dst_type)] = "{}"  # 非疾病实体暂无属性

                # 正向关系
                relations.append((disease_name, rel_type, val, weight))

                # 反向边: 症状 → 可能病因 → 疾病
                if json_field == "symptom":
                    relations.append((val, REVERSE_REL[0], disease_name, REVERSE_REL[3]))

    return entities, relations, dirty_deleted, dirty_rel_impact


# ============================================================
# SQLite 批量写入
# ============================================================

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS kg_entities (
    name TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    properties TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_kg_entities_type ON kg_entities(type);

CREATE TABLE IF NOT EXISTS kg_relations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    src TEXT NOT NULL,
    rel_type TEXT NOT NULL,
    dst TEXT NOT NULL,
    weight REAL DEFAULT 1.0,
    confidence REAL DEFAULT 1.0,
    UNIQUE(src, rel_type, dst)
);
CREATE INDEX IF NOT EXISTS idx_kg_relations_src ON kg_relations(src);
CREATE INDEX IF NOT EXISTS idx_kg_relations_dst ON kg_relations(dst);

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
"""


def write_to_db(db_path, entities, relations):
    """批量写入 SQLite(executemany + 事务)"""
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA_SQL)

    # 清空旧数据(重新导入)
    conn.execute("DELETE FROM kg_entities")
    conn.execute("DELETE FROM kg_relations")
    conn.execute("DELETE FROM kg_pagerank")
    conn.execute("DELETE FROM communities")

    # 批量写入实体
    conn.executemany(
        "INSERT OR IGNORE INTO kg_entities (name, type, properties) VALUES (?, ?, ?)",
        [(name, etype, props) for (name, etype), props in entities.items()]
    )

    # 批量写入关系
    conn.executemany(
        "INSERT OR IGNORE INTO kg_relations (src, rel_type, dst, weight, confidence) VALUES (?, ?, ?, ?, 1.0)",
        [(src, rel, dst, w) for src, rel, dst, w in relations]
    )

    conn.commit()
    conn.close()

    return len(entities), len(relations)


# ============================================================
# PageRank (networkx)
# ============================================================

def compute_pagerank(db_path, sample_info=None):
    """从 SQLite 读取关系,构建 networkx DiGraph,计算 PageRank"""
    import networkx as nx

    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT src, rel_type, dst, weight FROM kg_relations").fetchall()
    conn.close()

    G = nx.DiGraph()
    for src, rel_type, dst, weight in rows:
        G.add_edge(src, dst, rel_type=rel_type, weight=weight)

    print(f"  图节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()}")

    start = time.time()
    pr = nx.pagerank(G, alpha=0.5, weight='weight')
    elapsed = time.time() - start

    print(f"  PageRank 计算耗时: {elapsed:.2f}s")

    # 写入 kg_pagerank 表
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM kg_pagerank")
    conn.executemany(
        "INSERT OR REPLACE INTO kg_pagerank (name, score) VALUES (?, ?)",
        [(name, score) for name, score in pr.items()]
    )
    conn.commit()
    conn.close()

    return elapsed, len(pr)


# ============================================================
# 社区检测 (igraph + leidenalg)
# ============================================================

def compute_communities(db_path, resolutions=[0.5, 1.0, 1.5]):
    """从 SQLite 读取关系,构建 igraph Graph,用 Leiden 做三层社区检测"""
    import igraph as ig
    import leidenalg

    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT src, dst, weight FROM kg_relations").fetchall()
    conn.close()

    # 构建 igraph 无向图(社区检测用无向)
    # 先收集所有唯一节点名,再用索引映射
    node_names = sorted(set(r[0] for r in rows) | set(r[1] for r in rows))
    node_idx = {name: i for i, name in enumerate(node_names)}

    edges = [(node_idx[r[0]], node_idx[r[1]]) for r in rows]
    weights = [r[2] for r in rows]

    g = ig.Graph(n=len(node_names), edges=edges, directed=False)
    g.es["weight"] = weights

    print(f"  igraph 图节点数: {g.vcount()}, 边数: {g.ecount()}")

    # 三层 Leiden 社区检测
    communities_data = {}
    total_time = 0

    for level, resolution in enumerate(resolutions):
        start = time.time()
        partition = leidenalg.find_partition(
            g, leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=resolution, weights="weight"
        )
        elapsed = time.time() - start
        total_time += elapsed

        n_communities = len(partition)
        print(f"  Leiden L{level}(resolution={resolution}): {n_communities} 个社区, 耗时 {elapsed:.2f}s")

        # 收集社区成员
        level_data = []
        for cid, members in enumerate(partition):
            member_names = [node_names[v] for v in members]
            level_data.append({
                "community_id": f"L{level}_C{cid:03d}",
                "members": member_names,
            })
        communities_data[f"L{level}"] = level_data

    print(f"  社区检测总耗时: {total_time:.2f}s")

    # 对每个社区生成摘要
    from src.graphrag import CommunitySummaryGenerator
    gen = CommunitySummaryGenerator()

    # 也需要关系数据来生成摘要
    rel_dict = {}
    for src, dst, weight in rows:
        rel_dict.setdefault(src, []).append({"src": src, "dst": dst, "weight": weight})

    summary_start = time.time()
    community_rows = []
    for level_key, level_data in communities_data.items():
        level_num = int(level_key[1])
        for comm in level_data:
            members = comm["members"]
            if len(members) < 2:
                continue
            members_set = set(members)
            # 收集社区内关系
            intra_rels = []
            for m in members:
                for r in rel_dict.get(m, []):
                    if r["dst"] in members_set:
                        intra_rels.append(r)

            result = gen.generate(comm["community_id"], level_num, members, intra_rels)
            community_rows.append((
                result.community_id,
                result.level,
                result.theme,
                json.dumps(members, ensure_ascii=False),
                result.narrative,
                json.dumps(members[:10], ensure_ascii=False),  # match_keywords 用前 10 个成员
            ))

    summary_elapsed = time.time() - summary_start
    print(f"  摘要生成耗时: {summary_elapsed:.2f}s ({len(community_rows)} 个社区)")

    # 写入 communities 表
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM communities")
    conn.executemany(
        "INSERT OR REPLACE INTO communities (community_id, level, theme, core_entities, summary, match_keywords) VALUES (?, ?, ?, ?, ?, ?)",
        community_rows
    )
    conn.commit()
    conn.close()

    return total_time + summary_elapsed


# ============================================================
# 2-3 字症状词频次清单(供人工审查人名)
# ============================================================

def print_symptom_word_frequency(records):
    """把 symptom 字段里所有 2-3 个汉字的词,按出现频次排序全部打印"""
    from collections import Counter
    freq = Counter()

    for r in records:
        symptoms = r.get("symptom", [])
        if isinstance(symptoms, str):
            symptoms = [symptoms] if symptoms else []
        for s in symptoms:
            s = s.strip()
            if s and len(s) >= 2 and len(s) <= 3:
                # 只统计纯中文(不含英文/数字/符号)
                if re.match(r'^[一-鿿]{2,3}$', s):
                    freq[s] += 1

    print(f"\n=== 2-3字纯中文症状词频次清单(共 {len(freq)} 个词) ===")
    print("格式: 词 → 出现次数")
    print("-" * 50)
    # 按频次降序排列
    for word, count in freq.most_common():
        print(f"  {word} → {count}")


# ============================================================
# 主流程
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(ROOT / "data" / "medical_agent.db"))
    parser.add_argument("--json", default=str(ROOT / "data" / "medical_new_2.json"))
    parser.add_argument("--sample", type=int, default=None,
                        help="只用前 N 条记录做小样本测试(测耗时)")
    parser.add_argument("--skip-pagerank", action="store_true", help="跳过 PageRank 计算")
    parser.add_argument("--skip-community", action="store_true", help="跳过社区检测")
    parser.add_argument("--freq-only", action="store_true", help="只打印症状词频次清单,不导入")
    args = parser.parse_args()

    print("=" * 60)
    print("  医疗 KG 数据导入")
    print("=" * 60)

    # 读取 JSON
    print("\n[1] 读取 JSON 数据...")
    records = load_records(args.json)
    print(f"  总记录数: {len(records)}")

    # 打印 2-3 字症状词频次清单
    print_symptom_word_frequency(records)

    if args.freq_only:
        print("\n--freq-only 模式,只打印清单,不导入。")
        return

    # 抽取实体和关系
    print("\n[2] 抽取实体和关系...")
    sample_label = f"(sample={args.sample})" if args.sample else "(全量)"
    entities, relations, dirty_deleted, dirty_rel_impact = extract_entities_and_relations(
        records, sample_size=args.sample
    )
    print(f"  实体数: {len(entities)} {sample_label}")
    print(f"  关系数: {len(relations)} {sample_label}")

    # 脏数据报告
    print(f"\n[脏数据报告]")
    print(f"  删除的实体数: {len(dirty_deleted)}")
    print(f"  连带删除的关系数: {dirty_rel_impact}")
    if dirty_deleted:
        print(f"  删除的具体词:")
        for name, reason in sorted(dirty_deleted.items()):
            print(f"    \"{name}\" → {reason}")

    # 写入 SQLite
    print("\n[3] 写入 SQLite...")
    start = time.time()
    n_ent, n_rel = write_to_db(args.db, entities, relations)
    elapsed = time.time() - start
    print(f"  写入 {n_ent} 实体, {n_rel} 关系, 耗时 {elapsed:.2f}s")

    # PageRank
    if not args.skip_pagerank:
        print("\n[4] 计算 PageRank (networkx, alpha=0.5)...")
        pr_time, n_pr = compute_pagerank(args.db, sample_info=args.sample)
    else:
        print("\n[4] 跳过 PageRank (--skip-pagerank)")

    # 社区检测
    if not args.skip_community:
        print("\n[5] 社区检测 + 摘要生成 (igraph + leidenalg)...")
        comm_time = compute_communities(args.db)
    else:
        print("\n[5] 跳过社区检测 (--skip-community)")

    print("\n" + "=" * 60)
    print("  导入完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()