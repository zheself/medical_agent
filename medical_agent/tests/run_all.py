"""
tests/run_all.py — 一键跑所有单元测试

    cd medical_agent
    python -m tests.run_all

无需 pytest。装了 pytest 后也可直接 `pytest tests/`。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests import test_p0, test_core, test_database


def main():
    print("=" * 60)
    print("  运行全部单元测试")
    print("=" * 60)

    print("\n[1/3] tests/test_p0.py — P0 改动")
    ok1 = test_p0._run_all()

    print("\n[2/3] tests/test_core.py — 核心算法 + 开关")
    ok2 = test_core._run_all()

    print("\n[3/3] tests/test_database.py — SQLite .db 后端")
    ok3 = test_database._run_all()

    print("\n" + "=" * 60)
    if ok1 and ok2 and ok3:
        print("  ✅ 全部测试通过")
        return 0
    print("  ❌ 存在失败用例")
    return 1


if __name__ == "__main__":
    sys.exit(main())
