"""Build the deterministic V10c episodic-memory benchmark (60 scenarios)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parent.parent

ALLERGY_CASES = [
    ("青霉素", "阿莫西林"), ("头孢", "头孢克洛"), ("布洛芬", "布洛芬"),
    ("阿司匹林", "阿司匹林"), ("磺胺", "复方磺胺甲噁唑"),
    ("甲硝唑", "甲硝唑"), ("青霉素", "氨苄西林"), ("头孢", "头孢呋辛"),
    ("布洛芬", "芬必得"), ("阿司匹林", "拜阿司匹灵"),
]

CHRONIC_CASES = [
    ("糖尿病", "最近视力模糊，需要注意什么？"),
    ("高血压", "最近持续头痛，用药时要注意什么？"),
    ("哮喘", "最近咳嗽气短，应该注意什么？"),
    ("冠心病", "今天胸痛心悸，需要怎么处理？"),
    ("慢性肾病", "最近乏力恶心，用药要注意什么？"),
    ("糖尿病", "最近口渴多尿，可能与既往病史有关吗？"),
    ("高血压", "测得血压升高并头晕，既往情况重要吗？"),
    ("哮喘", "运动后呼吸困难，既往疾病是否相关？"),
    ("冠心病", "活动后胸闷，是否需要立即就医？"),
    ("慢性肾病", "出现水肿乏力，既往病史有什么影响？"),
]

MEDICATION_CASES = [
    ("二甲双胍", "糖尿病", "最近腹泻恶心，和正在吃的药有关吗？"),
    ("胰岛素", "糖尿病", "今天头晕乏力，是否需要考虑当前用药？"),
    ("氨氯地平", "高血压", "最近下肢肿胀，和长期用药有关吗？"),
    ("硝苯地平", "高血压", "服药后头痛心悸，应该怎么办？"),
    ("阿司匹林", "冠心病", "出现消化道出血迹象，当前用药重要吗？"),
    ("二甲双胍", "糖尿病", "准备做增强检查，需要说明哪些长期用药？"),
    ("胰岛素", "糖尿病", "没吃饭但已经用药，现在心悸出汗怎么办？"),
    ("布洛芬", "偏头痛", "最近胃痛恶心，是否与止痛药有关？"),
    ("甲硝唑", "胃炎", "治疗期间饮酒后恶心心悸，应注意什么？"),
    ("对乙酰氨基酚", "流感", "退烧药用了几天仍发热，下一步怎么办？"),
]

HISTORY_CASES = [
    ("偏头痛", "既往反复偏头痛", "再次出现单侧头痛畏光，既往诊断有帮助吗？"),
    ("肺炎", "上月确诊肺炎", "现在又咳嗽发热，既往诊断是否相关？"),
    ("胃炎", "胃镜提示慢性胃炎", "最近腹痛恶心，需要结合哪项既往检查？"),
    ("肝炎", "既往检查提示肝炎", "现在出现黄疸乏力，历史结果重要吗？"),
    ("视网膜病变", "眼底检查提示视网膜病变", "糖尿病并视力模糊，应结合什么既往结果？"),
    ("脑膜炎", "既往因脑膜炎住院", "现在发热头痛颈部僵硬，既往诊断是否重要？"),
    ("冠心病", "冠脉检查提示冠心病", "活动后胸痛，历史检查如何影响判断？"),
    ("高血压", "动态血压提示高血压", "今天头晕头痛，既往检查是否相关？"),
    ("糖尿病", "糖化血红蛋白支持糖尿病", "最近多饮多尿，应参考哪项既往结果？"),
    ("哮喘", "肺功能检查支持哮喘", "接触冷空气后气短咳嗽，既往结果重要吗？"),
]

TEMPORAL_CASES = [
    ("青霉素过敏", "复查确认青霉素不过敏", "最新检查后，我还能否按旧的青霉素过敏史处理？", "不过敏"),
    ("正在服用二甲双胍", "医生已停用二甲双胍", "我现在还在使用二甲双胍吗？", "停用"),
    ("确诊高血压", "复诊认为暂不能诊断高血压", "我的最新高血压诊断状态是什么？", "暂不能诊断"),
    ("胰岛素每日使用", "方案调整为停用胰岛素", "当前是否还应按每日胰岛素方案处理？", "停用"),
    ("头孢过敏", "过敏专科复核未发现头孢过敏", "最新记录是否仍支持头孢过敏？", "未发现"),
    ("既往肺炎未愈", "复查显示肺炎已经痊愈", "目前肺炎是否仍处于未愈状态？", "痊愈"),
    ("长期服用阿司匹林", "因出血风险已停用阿司匹林", "现在是否仍在服用阿司匹林？", "停用"),
    ("糖尿病控制不佳", "最新复查显示血糖控制达标", "当前血糖控制状态应参考哪条记录？", "达标"),
    ("慢性胃炎活动期", "复查提示胃炎已缓解", "目前还应认为胃炎处于活动期吗？", "缓解"),
    ("视网膜病变进展", "最新眼底检查提示病变稳定", "当前视网膜病变状态是什么？", "稳定"),
]

IRRELEVANT_CASES = [
    "今天扭伤脚踝，应该怎么处理？", "最近失眠，应该注意什么？",
    "皮肤出现轻微擦伤，需要消毒吗？", "准备体检，通常需要空腹吗？",
    "最近肩膀酸痛，可能是什么原因？", "牙龈出血应该挂什么科？",
    "耳鸣持续一周，需要做什么检查？", "眼睛干涩，日常如何缓解？",
    "运动后肌肉酸痛是否正常？", "轻微鼻塞但不发热，需要吃药吗？",
]

NOISE = [
    {"summary": "既往确诊糖尿病，长期服用二甲双胍", "diagnoses": ["糖尿病"], "medications": ["二甲双胍"]},
    {"summary": "既往高血压，长期服用氨氯地平", "diagnoses": ["高血压"], "medications": ["氨氯地平"]},
    {"summary": "上次因流感发热咳嗽就诊", "diagnoses": ["流感"], "symptoms": ["发热", "咳嗽"]},
]


def _memory(episode_id: str, summary: str, *, scope: str = "target", days_ago: int = 10,
            diagnoses: List[str] | None = None, medications: List[str] | None = None,
            symptoms: List[str] | None = None) -> Dict[str, Any]:
    return {
        "episode_id": episode_id, "scope": scope, "summary": summary,
        "days_ago": days_ago, "diagnoses": diagnoses or [],
        "medications": medications or [], "symptoms": symptoms or [],
        "importance_score": 0.8,
    }


def _noise_memories(sid: str) -> List[Dict[str, Any]]:
    return [
        _memory(f"noise::{sid}::{index}", item["summary"], days_ago=20 + index,
                diagnoses=item.get("diagnoses"), medications=item.get("medications"),
                symptoms=item.get("symptoms"))
        for index, item in enumerate(NOISE[:2], start=1)
    ]


def build_scenarios() -> List[Dict[str, Any]]:
    scenarios: List[Dict[str, Any]] = []

    for index, (allergen, drug) in enumerate(ALLERGY_CASES, start=1):
        sid = f"allergy_{index:02d}"
        relevant = f"rel::{sid}"
        memories = [_memory(relevant, f"患者明确记录{allergen}严重过敏", diagnoses=["药物过敏"])]
        memories += _noise_memories(sid)
        memories.append(_memory(f"other::{sid}", f"另一患者记录{allergen}严重过敏", scope="other"))
        scenarios.append({
            "scenario_id": sid, "category": "allergy",
            "query": f"我现在出现咽痛发热，可以使用{drug}吗？",
            "memories": memories, "expected_memory_ids": [relevant],
            "forbidden_memory_ids": [], "expected_critical_facts": [allergen],
            "must_include": ["过敏"], "must_not_include": [],
        })

    for index, (disease, query) in enumerate(CHRONIC_CASES, start=1):
        sid = f"chronic_{index:02d}"
        relevant = f"rel::{sid}"
        memories = [_memory(relevant, f"患者长期患有{disease}", diagnoses=[disease], days_ago=180)]
        memories += _noise_memories(sid)
        memories.append(_memory(f"other::{sid}", f"另一患者长期患有{disease}", scope="other", diagnoses=[disease]))
        scenarios.append({
            "scenario_id": sid, "category": "chronic", "query": query,
            "memories": memories, "expected_memory_ids": [relevant],
            "forbidden_memory_ids": [], "expected_critical_facts": [disease],
            "must_include": [disease], "must_not_include": [],
        })

    for index, (drug, disease, query) in enumerate(MEDICATION_CASES, start=1):
        sid = f"medication_{index:02d}"
        relevant = f"rel::{sid}"
        memories = [_memory(relevant, f"因{disease}正在长期服用{drug}", diagnoses=[disease], medications=[drug])]
        memories += _noise_memories(sid)
        memories.append(_memory(f"other::{sid}", f"另一患者正在服用{drug}", scope="other", medications=[drug]))
        scenarios.append({
            "scenario_id": sid, "category": "medication", "query": query,
            "memories": memories, "expected_memory_ids": [relevant],
            "forbidden_memory_ids": [], "must_include": [drug], "must_not_include": [],
        })

    for index, (diagnosis, summary, query) in enumerate(HISTORY_CASES, start=1):
        sid = f"history_{index:02d}"
        relevant = f"rel::{sid}"
        memories = [_memory(relevant, summary, diagnoses=[diagnosis], days_ago=60)]
        memories += _noise_memories(sid)
        memories.append(_memory(f"other::{sid}", f"另一患者{summary}", scope="other", diagnoses=[diagnosis]))
        scenarios.append({
            "scenario_id": sid, "category": "history", "query": query,
            "memories": memories, "expected_memory_ids": [relevant],
            "forbidden_memory_ids": [], "must_include": [diagnosis], "must_not_include": [],
        })

    for index, (old_fact, new_fact, query, expected_term) in enumerate(TEMPORAL_CASES, start=1):
        sid = f"temporal_{index:02d}"
        old_id, new_id = f"old::{sid}", f"rel::{sid}"
        memories = [
            _memory(old_id, old_fact, days_ago=365),
            _memory(new_id, new_fact, days_ago=1),
            *_noise_memories(sid)[:1],
            _memory(f"other::{sid}", old_fact, scope="other", days_ago=1),
        ]
        scenarios.append({
            "scenario_id": sid, "category": "temporal", "query": query,
            "memories": memories, "expected_memory_ids": [new_id],
            "forbidden_memory_ids": [old_id], "must_include": [expected_term],
            "must_not_include": ["存在矛盾", "需要进一步确认", "无法确定"],
        })

    for index, query in enumerate(IRRELEVANT_CASES, start=1):
        sid = f"irrelevant_{index:02d}"
        memories = _noise_memories(sid)
        memories.append(_memory(f"other::{sid}", "另一患者有与当前问题相似的记录", scope="other"))
        scenarios.append({
            "scenario_id": sid, "category": "irrelevant", "query": query,
            "memories": memories, "expected_memory_ids": [],
            "forbidden_memory_ids": [m["episode_id"] for m in memories if m["scope"] == "target"],
            "must_include": [], "must_not_include": [],
        })

    assert len(scenarios) == 60
    return scenarios


def main() -> None:
    output = ROOT / "data" / "eval_memory_scenarios.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        for scenario in build_scenarios():
            handle.write(json.dumps(scenario, ensure_ascii=False) + "\n")
    print(f"Wrote 60 scenarios to {output}")


if __name__ == "__main__":
    main()
