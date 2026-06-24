"""
tests/test_database.py — SQLite .db 后端测试

覆盖:
- 数据库种子生成（建表 + 填数据）
- SQLiteKGBackend 查询邻居 / PageRank
- factory 的 db 模式构建 + 端到端
- db 模式真实读取用户历史（过敏史/病史注入）

无需 pytest:
    cd medical_agent
    python -m tests.test_database

注意: 本测试会在临时目录生成 .db，不污染 data/medical_agent.db。
"""
from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.schemas import UserQuery


def _make_temp_db() -> str:
    """生成一个临时数据库，返回路径"""
    from scripts.seed_database import seed
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    seed(tmp.name, force=True, quiet=True)
    return tmp.name


# ============================================================
# 种子数据库
# ============================================================

def test_seed_creates_all_tables():
    """种子脚本应创建所有预期的表并填入数据"""
    import sqlite3
    db = _make_temp_db()
    conn = sqlite3.connect(db)
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    expected = {"kg_entities", "kg_relations", "kg_pagerank",
                "communities", "episodes", "failure_cases", "dynamic_rules"}
    assert expected <= tables, f"缺少表: {expected - tables}"
    # 各表非空
    for t in ["kg_entities", "kg_relations", "kg_pagerank", "episodes"]:
        n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        assert n > 0, f"{t} 为空"
    conn.close()


# ============================================================
# SQLiteKGBackend
# ============================================================

def test_sqlite_kg_query_neighbors():
    """查询脑膜炎邻居应返回其典型症状"""
    from src.tools.kg_local_search import SQLiteKGBackend
    db = _make_temp_db()
    kg = SQLiteKGBackend(db)
    nb = asyncio.run(kg.query_neighbors("脑膜炎"))
    targets = {f["target"] for f in nb}
    assert "颈强直" in targets and "头痛" in targets
    # 结构字段齐全
    assert all("rel" in f and "weight" in f and "target_type" in f for f in nb)


def test_sqlite_kg_pagerank():
    """PageRank 应返回合理分数，未知节点返回默认值"""
    from src.tools.kg_local_search import SQLiteKGBackend
    db = _make_temp_db()
    kg = SQLiteKGBackend(db)
    pr_known = asyncio.run(kg.get_pagerank("脑膜炎"))
    pr_unknown = asyncio.run(kg.get_pagerank("不存在的实体"))
    assert pr_known > pr_unknown
    assert pr_unknown == 0.01


def test_sqlite_kg_missing_db_raises():
    """指向不存在的 .db 应抛 FileNotFoundError"""
    from src.tools.kg_local_search import SQLiteKGBackend
    try:
        SQLiteKGBackend("/nonexistent/path/x.db")
        assert False, "应抛异常"
    except FileNotFoundError:
        pass


def test_ppr_on_sqlite_kg():
    """PPR 在 SQLite KG 上应识别脑膜炎为最相关"""
    from src.tools.kg_local_search import SQLiteKGBackend
    from src.tools.ppr_reasoner import PPRReasonerTool
    db = _make_temp_db()
    kg = SQLiteKGBackend(db)
    tool = PPRReasonerTool(kg_backend=kg, alpha=0.5)
    res = asyncio.run(tool.ainvoke({"seed_entities": ["头痛", "发烧", "颈强直"], "top_k": 5}))
    assert res.success
    assert res.data["relevant_concepts"][0]["entity"] == "脑膜炎"


# ============================================================
# factory db 模式
# ============================================================

def test_factory_db_mode_builds():
    """factory 的 db 模式应能构建系统"""
    from src.factory import build_system
    db = _make_temp_db()
    orch, episodic = build_system(backend="db", db_path=db)
    assert orch is not None and episodic is not None


def test_factory_db_injects_user_history():
    """db 模式下，已有历史的用户应被注入过敏史和病史"""
    from src.factory import build_system
    db = _make_temp_db()
    orch, _ = build_system(backend="db", db_path=db)
    # patient_001 在种子数据里有糖尿病 + 青霉素过敏
    ans = asyncio.run(orch.answer_async(
        UserQuery(user_id="patient_001", text="我头痛"), session_id="s_db"))
    wm = orch.wm_pool["s_db"]
    assert "青霉素" in wm.patient_profile.allergies, f"应注入过敏史: {wm.patient_profile.allergies}"
    assert any("糖尿病" in d for d in wm.patient_profile.medical_history)


def test_factory_mock_and_db_same_interface():
    """两种后端返回一致的接口"""
    from src.factory import build_system
    db = _make_temp_db()
    o1, e1 = build_system(backend="mock")
    o2, e2 = build_system(backend="db", db_path=db)
    for o in (o1, o2):
        assert hasattr(o, "answer_async")
        assert hasattr(o, "enable_prefetch")
    for e in (e1, e2):
        assert hasattr(e, "retrieve")


# ============================================================
# Runner
# ============================================================

def _run_all():
    passed, failed = 0, 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ✅ {name}")
                passed += 1
            except Exception as e:
                print(f"  ❌ {name}: {type(e).__name__}: {e}")
                failed += 1
    print(f"\n{'='*50}\n通过 {passed} / {passed + failed}")
    return failed == 0


if __name__ == "__main__":
    print("Running database tests...\n")
    sys.exit(0 if _run_all() else 1)
