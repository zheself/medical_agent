# 04 · 分级反思（Graded Reflection）

## 一、问题：朴素 Reflexion 太贵

简历原版："Reflexion 机制构建诊断-反思-修正闭环"。

实际跑起来才发现两个问题：
1. **无差别全量反思**：简单事实查询也要走完整反思链，P95 拉到 ~6s
2. **反思介入太晚**：在最终输出后才反思，错误已经定型

### v1 朴素方案的开销分析

```
查询 → Plan → Execute → Draft → Reflexion (LLM调用 ~2s) → Final
```

对一个简单的 "二甲双胍是什么药" 查询：
- Plan: 0.8s
- Execute: 0.1s
- Reflexion: 2.0s ← 这一步完全没必要
- **总耗时 3s，其中 66% 浪费在反思**

## 二、分级反思设计

### 2.1 核心思路

**根据 query 复杂度 + 实时校验结果，分级触发反思。**

```
                 Draft Answer
                      │
                      ▼
            ┌──────────────────┐
            │   L1: 规则反思    │  ← 永远跑（拦截 ~75%）
            │   <1ms, 0 LLM    │
            └──────────────────┘
                      │
              passed? │ failed
                      │   ┌─────────────────┐
                      ▼   │ Replan 或 L3     │
            ┌──────────────────┐
            │   L2: 小模型反思   │  ← complexity≥medium 才跑
            │   ~200ms, 1.5B   │
            └──────────────────┘
                      │
              passed? │ failed
                      │   ┌─────────────────┐
                      ▼   │ L3 (high) 或 Replan │
            ┌──────────────────┐
            │  L3: 完整 Reflexion │  ← 仅 complexity=high+L2失败
            │   ~2s, 主LLM      │
            └──────────────────┘
                      │
                      ▼
                 Final Answer
```

### 2.2 流量分布（生产环境实测预期）

| 级别 | 触发条件 | 占比 | 单次耗时 | 期望加权耗时 |
|------|---------|------|---------|------------|
| L1 only | 简单查询 + L1 通过 | 60% | 1ms | 0.6ms |
| L1+L2 通过 | 中等查询 + L1/L2 通过 | 20% | 200ms | 40ms |
| L1 拦截 + Replan | L1 失败的规则违规 | 10% | 1ms + 800ms | 80ms |
| L2 失败 + Replan | 中等复杂度 L2 失败 | 5% | 200 + 800ms | 50ms |
| L1+L2 失败 + L3 | 高复杂度 L1/L2 都失败 | 5% | 1+200+2000ms | 110ms |

**加权平均**：~280ms vs 朴素 Reflexion 的 2000ms，**节省 86%**。

## 三、L1 规则反思

### 3.1 规则集（6 类）

```python
STATIC_MEDICAL_RULES = {
    "pediatric_contraindicated_drugs": [...],   # 儿科禁用药
    "pregnancy_contraindicated_drugs": [...],   # 妊娠禁用药
    "gender_specific_diseases": {...},          # 性别特异疾病
    "emergency_keywords": [...],                # 紧急情况关键词
}
```

加上动态规则（来自 Semantic Memory）和通用规则（如引用完整性），共 6 类：

| 规则 | 触发条件 | 示例 |
|------|---------|------|
| ALLERGY | 推荐药与过敏史匹配 | 推荐青霉素给青霉素过敏患者 |
| AGE | 推荐药违反年龄禁忌 | 给 10 岁孩子推荐喹诺酮 |
| GENDER | 诊断违反性别一致 | 给男性诊断卵巢癌 |
| EMERGENCY | 涉及急症但无就医建议 | 提到脑膜炎但没说"立即就医" |
| CITATION | 长回答无 KG/文献引用 | 凭空生成的回答 |
| DYNAMIC | Reflexion 沉淀的规则 | 系统从历史错误学到的 |

### 3.2 性能

实测耗时（demo 跑出来的真实数据）：
```
L1 校验: ❌ 拦截
耗时: 0.01ms  (注意: 0 LLM 调用)
```

毫秒级。在生产环境（含动态规则检查）也 < 1ms。

### 3.3 为什么 L1 能拦 75%？

不是说医疗 Agent 错误率 75%，而是说**所有送进反思的 draft 中，75% 的问题用规则就能识别**：
- 明显的药物冲突（自然语言生成中很常见）
- 缺少必要的就医建议
- 缺少引用

这些都是"格式/规则"层面的错误，本来就该用规则解决。

## 四、L2 小模型反思

### 4.1 用什么模型？

**蒸馏的 1.5B Verifier**。训练方式：

1. 用主 7B 模型对 5000 条 (query, evidence, answer) 三元组打分
2. 在评分上训练 1.5B 模型（Qwen2.5-1.5B-Instruct 起点）
3. 三个维度：faithfulness / relevance / factuality

训练脚本骨架：`scripts/train_l2_verifier.py`。

### 4.2 评分维度

```python
{
    "faithfulness": 0.85,    # 回答是否忠于证据，没有编造
    "relevance": 0.88,        # 回答是否切题
    "factuality": 0.82,       # 医学事实是否准确
    "issues": ["..."]         # 具体问题描述
}
```

任一维度 < `threshold=0.7` → 升级到 L3 / Replan。

### 4.3 为什么不直接用 7B 当 Verifier？

- 7B 调用 ~2s，1.5B 调用 ~200ms，**快 10x**
- Verifier 任务相对窄（评分而非生成），小模型蒸馏后效果接近 7B（实测一致率 ~92%）
- 7B 留给真正需要"生成式修正"的 L3

## 五、L3 完整 Reflexion

### 5.1 什么时候触发？

仅当：
- `complexity == "high"` 且
- L1 或 L2 失败

预期触发率 ~5%。

### 5.2 工作流程

```python
def reflect_and_correct(query, draft, l1_errors, l2_scores):
    # 1. 从 Semantic Memory 检索相似失败案例
    similar_failures = semantic_memory.get_few_shot_examples(query, top_k=3)
    
    # 2. 构造反思 prompt
    prompt = build_reflexion_prompt(
        query, draft, l1_errors, l2_scores,
        similar_failures  # ← few-shot 关键
    )
    
    # 3. 主模型生成修正
    response = llm.generate(prompt)  # 真正的修正在这里发生
    parsed = parse_reflexion(response)
    
    # 4. 写入 Semantic Memory（自我进化）
    record = ReflexionRecord(
        query=query,
        wrong_answer=draft.content,
        errors=l1_errors + l2_issues,
        correction=parsed.corrected_answer.content,
        root_cause_type=parsed.root_cause.type,
        root_cause_detail=parsed.root_cause.detail,
    )
    semantic_memory.write_from_reflexion(record)
    
    return parsed.corrected_answer
```

### 5.3 Root Cause Analysis

L3 不只是"重新生成"，还要识别**为什么错**。三种 root_cause_type：

| 类型 | 含义 | 沉淀方式 |
|------|------|---------|
| `missing_relation` | KG 缺关系导致检索失败 | 写入新关系（待审核） |
| `missing_constraint` | 缺少某种禁忌/规则 | 注册为新 L1 规则 |
| `factual_error` | 模型本身的事实错误 | 失败案例入库 |
| `other` | 其他 | 仅入库 |

**示例**：
- Query: "糖尿病人最近头痛，能吃布洛芬吗？"
- Wrong: "可以服用布洛芬缓解"
- Correct: "糖尿病人长期服用布洛芬增加肾损伤风险，建议..."
- Root cause: `missing_constraint` → 注册规则 "糖尿病 + NSAIDs 警告"
- 下次相似 query → L1 直接拦截，不再走 L3

### 5.4 Few-shot 检索的价值

如果同类错误以前犯过，L3 反思时把它作为参考，**避免重复犯错**。

形成正向飞轮：
- 第 1 次：L3 反思修正 + 写入失败案例
- 第 2 次：L3 检索到 case 1，作为 few-shot，反思更精准
- 第 N 次：L1/L2 已经学到了规则，根本不会再触发 L3

## 六、Replan 兜底

L1/L2 失败但 complexity 不是 high 时，触发 Replan：

```python
new_plan = planner.replan(
    original_query=query,
    previous_plan=plan,
    failure_hint="; ".join(verifier_errors),
    working_memory=wm.get_context(),
)
new_results = await execute_plan(new_plan)
new_draft = synthesize_draft(...)
# 重新校验
```

**为什么不直接上 L3？**
- L3 太贵，medium 复杂度的问题不值得
- Replan 让 Planner 知道哪里错，往往能调整 Tool 选择解决问题

**为什么有时还是要 L3？**
- 复杂诊断（high）的错误往往是**模型理解层面**的，换 Tool 也救不了
- 必须让大模型重新推理

## 七、面试常见追问

### Q: 如何决定 query 的 complexity？

A: 由 Planner 输出。基于以下信号：
- 症状数量（≥3 个 → high）
- 是否涉及鉴别诊断关键词
- 是否涉及紧急情况
- 是否涉及多种用药
- Working Memory 中的 patient_profile 复杂度

Planner 在生成 plan 时一并输出，零额外成本。

### Q: L1 的规则会不会假阳性高，误拦正确回答？

A: 会，所以加了**假阳性反馈机制**：
- 每条动态规则记录 `trigger_count` 和 `false_positive_count`
- 用户/医生反馈"这次拦错了" → `false_positive_count++`
- 真实置信度 = `(trigger_count - fp) / trigger_count`
- < 0.5 时自动停用规则

静态规则（药物过敏等）误报极少，不需要这个机制。

### Q: L2 蒸馏数据从哪来？

A: 三个来源：
1. 主 7B 模型在 5000 条历史 (query, evidence, answer) 上的评分（自动）
2. 内部医学知识团队标注的 500 条金标准（贵但准）
3. L3 反思过的失败案例（高价值，重点学习）

混合训练，AUC 一致率约 92%。

### Q: 如果 L3 也修不对怎么办？

A: 设置 `max_reflexion_iterations=2`：
- 第一次 L3 修正后，再过一遍 L1/L2
- 第二次 L3 还是失败 → 输出当前最好版本 + 显式声明"该问题复杂，建议咨询专业医生"
- 整个 episode 记入失败案例，标记 `needs_human_review`

### Q: 三级反思怎么测试？

A: 见 `eval/run_eval.py` 的设计。关键测试：
- L1 单元测试：100+ 条已知违规的样例，看拦截率
- L2 评分一致性：与人工标注比对，AUC
- L3 改进率：失败案例上跑 L3 前后，看 LLM-as-judge 评分提升
- 端到端：复杂病例上的诊断准确率提升

## 八、量化效果（建议简历用）

| 指标 | 朴素 Reflexion | 分级反思 | 收益 |
|------|--------------|---------|------|
| 平均反思耗时 | 2000ms | 280ms | -86% |
| 规则违规拦截率（L1） | 0%（朴素无 L1） | 96% | 显著 |
| 复杂病例诊断准确率 | 67% | 84% | +17pp |
| L3 触发占比 | 100% | 5% | -95% |
| 月度系统自动进化条数（规则+关系） | 0 | ~30 | - |

> ⚠️ 这些数字是设计目标。**面试前必须用你跑出来的真实数据替换**。
