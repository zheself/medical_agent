"""
app/streamlit_app.py — 医疗诊断 Agent 可视化界面

运行:
    cd medical_agent
    pip install streamlit
    streamlit run app/streamlit_app.py

界面展示项目的核心亮点:
- 多轮问诊对话
- Planner 生成的 DAG 计划
- 工具执行轨迹（NER / GraphRAG / PPR）
- 三层 Memory 状态（Working / Episodic）
- 分级反思链路（L1 / L2 / L3）

当前用 Mock 后端，无需 GPU / Neo4j / vLLM 即可运行。
切换真实后端见 README "切换到生产环境"。
"""
from __future__ import annotations

import asyncio
import sys
from dataclasses import asdict
from pathlib import Path

import streamlit as st

# 让 app 能 import src
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.schemas import UserQuery
from src.memory.working_memory import format_patient_card


# ============================================================
# 系统初始化（缓存，避免每次交互重建）
# ============================================================

@st.cache_resource
def get_system(backend: str = "mock"):
    """构建 Agent 系统。cache_resource 按 backend 缓存，全会话只建一次。"""
    from src.factory import build_system
    orch, episodic = build_system(backend=backend)
    return orch, episodic


def run_async(coro):
    """在 Streamlit 同步环境里跑异步协程"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():  # 极少见，兜底
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(coro)
    except RuntimeError:
        pass
    return asyncio.run(coro)


# ============================================================
# 页面配置
# ============================================================

st.set_page_config(
    page_title="医疗诊断 Agent",
    page_icon="🩺",
    layout="wide",
)

# 后端选择（mock 或 db）。db 模式需先运行 scripts/seed_database.py
import os as _os
_db_exists = _os.path.exists(str(ROOT / "data" / "medical_agent.db"))
_backend_choice = st.sidebar.radio(
    "🗄️ 数据后端",
    options=(["mock", "db"] if _db_exists else ["mock"]),
    format_func=lambda x: "全内存 Mock" if x == "mock" else "SQLite 真实库",
    help="db 模式从 data/medical_agent.db 读取真实数据；"
         + ("" if _db_exists else "（未检测到 .db，请先运行 scripts/seed_database.py）"),
)
orch, episodic = get_system(_backend_choice)

# 会话状态
if "history" not in st.session_state:
    st.session_state.history = []          # [(role, text)]
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None
if "session_id" not in st.session_state:
    st.session_state.session_id = "ui_session"
if "user_id" not in st.session_state:
    st.session_state.user_id = "ui_user"


# ============================================================
# 侧边栏：设置 + 消融开关 + 预置用户历史
# ============================================================

with st.sidebar:
    st.title("🩺 医疗诊断 Agent")
    st.caption("基于多 Agent 协作与知识图谱 · Mock 演示")

    st.subheader("⚙️ 系统开关")
    enable_prefetch = st.checkbox("推测式预取 (Speculative Pre-fetch)", value=True)
    enable_memory = st.checkbox("Memory 注入", value=True)
    max_level = st.select_slider(
        "反思最高级别",
        options=["L1", "L2", "L3"],
        value="L3",
        help="L1=仅规则 / L2=+小模型 / L3=+完整反思",
    )
    # 应用开关（真实生效）
    orch.enable_prefetch = enable_prefetch
    orch.enable_memory_injection = enable_memory
    orch.max_verifier_level = max_level
    if hasattr(orch.verifier, "max_level"):
        orch.verifier.max_level = max_level

    st.divider()
    st.subheader("👤 模拟老用户历史")
    st.caption("预置历史后，问诊时会触发 Episodic 检索与冷启动注入")
    if st.button("写入示例历史 (糖尿病 + 青霉素过敏)"):
        from src.memory.episodic_memory import Episode
        episodic.write(Episode(
            user_id=st.session_state.user_id,
            episode_type="diagnosis",
            diagnoses=["糖尿病"], medications=["二甲双胍"],
            symptoms=["多饮", "多尿"],
            summary="确诊2型糖尿病，服用二甲双胍，青霉素过敏",
        ))
        st.success("已写入示例历史")

    st.divider()
    if st.button("🔄 清空当前对话"):
        st.session_state.history = []
        st.session_state.last_answer = None
        # 清掉该 session 的 working memory
        orch.wm_pool.pop(st.session_state.session_id, None)
        st.rerun()

    st.divider()
    st.caption("⚠️ Mock 演示系统，回答不构成医疗建议。")


# ============================================================
# 主区域
# ============================================================

st.header("医疗诊断推理")

# —— 对话历史 ——
for role, text in st.session_state.history:
    with st.chat_message("user" if role == "user" else "assistant"):
        st.markdown(text)

# —— 输入 ——
query = st.chat_input("描述症状或提问，例如：头痛三天伴发烧，颈部僵硬，可能是什么病？")

if query:
    st.session_state.history.append(("user", query))
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Agent 推理中…"):
            answer = run_async(orch.answer_async(
                UserQuery(user_id=st.session_state.user_id, text=query),
                session_id=st.session_state.session_id,
            ))
        st.markdown(answer.content)
    st.session_state.history.append(("assistant", answer.content))
    st.session_state.last_answer = answer


# ============================================================
# 推理详情面板（展示项目内部机制）
# ============================================================

ans = st.session_state.last_answer
if ans is not None:
    st.divider()
    st.subheader("🔍 最近一次推理详情")

    # 顶部指标
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("复杂度", ans.plan.complexity.value if ans.plan else "-")
    c2.metric("耗时", f"{ans.total_elapsed_ms:.0f} ms")
    c3.metric("引用数", len(ans.citations))
    final_level = ans.verification_results[-1].level.value if ans.verification_results else "-"
    c4.metric("反思到达级别", final_level)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📋 Planner 计划", "🔧 执行轨迹", "🧠 Memory 状态", "✅ 分级反思"]
    )

    # —— Tab 1: Planner 计划 ——
    with tab1:
        if ans.plan:
            st.markdown(f"**Planner 思考**：{ans.plan.thought}")
            st.markdown(f"**复杂度判断**：`{ans.plan.complexity.value}`")
            st.markdown("**DAG 执行计划**：")
            for s in ans.plan.steps:
                deps = f" ← 依赖 {s.depends_on}" if s.depends_on else "（无依赖，可并行）"
                status_icon = "✅" if s.status == "done" else "❌"
                st.markdown(f"- {status_icon} **Step {s.step_id}**: `{s.tool}`{deps}")
            if ans.plan.speculative_prefetch:
                st.markdown(
                    f"**推测式预取候选**：{', '.join(ans.plan.speculative_prefetch)}"
                )
                st.caption(
                    f"预取命中率（累计）：{orch.get_prefetch_hit_rate():.0%} "
                    f"（{orch.prefetch_stats['total_hits']}/{orch.prefetch_stats['total_prefetched']}）"
                )

    # —— Tab 2: 执行轨迹 ——
    with tab2:
        if ans.plan:
            for s in ans.plan.steps:
                with st.expander(f"Step {s.step_id}: {s.tool}  ({s.elapsed_ms or 0:.1f} ms)"):
                    out = getattr(s, "output", None)
                    if out and getattr(out, "data", None):
                        data = out.data
                        # PPR 推理轨迹特别展示
                        if isinstance(data, dict) and "relevant_concepts" in data:
                            st.markdown("**PPR 多跳推理结果**（按相关性）：")
                            for item in data["relevant_concepts"][:8]:
                                st.markdown(
                                    f"- {item['entity']}：`{item['ppr_score']:.4f}`"
                                )
                        elif isinstance(data, dict) and "communities" in data:
                            st.markdown("**GraphRAG 社区检索**：")
                            for c in data["communities"]:
                                st.markdown(f"- **{c['theme']}**：{c['summary'][:120]}…")
                        elif isinstance(data, dict) and "facts" in data:
                            st.markdown("**KG 检索事实**：")
                            for f in data["facts"][:8]:
                                st.markdown(
                                    f"- {f.get('source','?')} —[{f.get('rel','?')}]→ "
                                    f"{f.get('target','?')}"
                                )
                        else:
                            st.json(data if isinstance(data, dict) else {"data": str(data)})
                    else:
                        st.caption("（无输出或执行失败）")

    # —— Tab 3: Memory 状态 ——
    with tab3:
        wm = orch.wm_pool.get(st.session_state.session_id)
        st.markdown("#### Working Memory（会话级）")
        if wm:
            st.code(format_patient_card(wm.patient_profile), language=None)
            colA, colB = st.columns(2)
            with colA:
                st.markdown("**已探索诊断**")
                st.write(list(wm.explored_diagnoses) or "（暂无）")
            with colB:
                st.markdown("**证据池计数**")
                st.write({k: len(v) for k, v in wm.retrieved_evidence.items()})
            if wm.compressed_summary:
                st.markdown("**压缩历史摘要**")
                st.caption(wm.compressed_summary[:300])
        else:
            st.caption("（本会话尚无 Working Memory）")

        st.markdown("#### Episodic Memory（用户级）")
        eps = episodic.backend.list_by_user(st.session_state.user_id)
        if eps:
            for e in eps:
                st.markdown(
                    f"- `{e.timestamp.strftime('%Y-%m-%d')}` "
                    f"**{'/'.join(e.diagnoses) or '咨询'}** "
                    f"(重要性 {e.importance_score:.2f}, 访问 {e.access_count}) — {e.summary[:50]}"
                )
        else:
            st.caption("（该用户暂无历史记录，可在侧边栏写入示例历史）")

    # —— Tab 4: 分级反思 ——
    with tab4:
        st.markdown("分级反思链路（L1 规则 → L2 小模型 → L3 完整反思，按需触发）：")
        for vr in ans.verification_results:
            icon = "✅" if vr.passed else "❌"
            with st.expander(
                f"{icon} {vr.level.value}  ({vr.elapsed_ms:.2f} ms)",
                expanded=not vr.passed,
            ):
                if vr.scores:
                    st.write({k: v for k, v in vr.scores.items() if isinstance(v, (int, float))})
                if vr.errors:
                    for err in vr.errors:
                        st.warning(err)
                if vr.passed and not vr.errors:
                    st.success("通过")
        st.caption(
            "💡 简单查询通常只在 L1 拦截（0 LLM 调用，毫秒级）；"
            "复杂病例才逐级升到 L2 / L3。"
        )


# ============================================================
# 页脚示例问题
# ============================================================

with st.expander("💬 试试这些问题"):
    st.markdown("""
    - `头痛三天伴发烧 38.5°C，颈部僵硬，可能是什么病？` （高复杂度 → 触发 GraphRAG + PPR + L3）
    - `二甲双胍是什么药？` （低复杂度 → 仅 L1）
    - `糖尿病人最近视力模糊，需要注意什么？` （中复杂度）
    - 先在侧边栏「写入示例历史」，再问 `我能吃青霉素吗？` （验证过敏拦截）
    """)
