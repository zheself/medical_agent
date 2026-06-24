"""
scripts/load_cmb_clin.py — CMB-Clin → EvalItem 格式适配

读取 CMB-Clin 原始 JSON，转成项目 EvalItem 格式的 JSONL。

query 构造：把 description（现病史+体格检查）拼成患者描述。
gold_diagnoses 提取：从 QA answer 中提取诊断名称（多策略正则匹配）。
difficulty 推断：根据症状数量和 title 中的科室分类。
min_hops 推断：多症状鉴别诊断=2，单症状=1。

用法:
    python scripts/load_cmb_clin.py
    # 输出: data/eval_cmb_clin.jsonl
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def extract_diagnoses(answer: str, title: str = '') -> list[str]:
    """从 CMB-Clin answer 中提取诊断名称（多策略匹配）"""
    diags = []

    # Strategy 1: "诊断：XXX" or "诊断:XXX"
    m = re.search(r'诊断[：:]\s*(.+)', answer)
    if m:
        raw = m.group(1).strip()

        # Detect whole-paragraph after "诊断：" — skip if starts with
        # narrative phrases like "本例病人" or is longer than 60 chars
        # with no numbered sub-items
        if (raw.startswith('本例') or raw.startswith('病人') or raw.startswith('该')) \
                and not re.search(r'[（(]\s*\d+[）)]', raw) \
                and not re.search(r'\d+[.、]\s*\S', raw):
            # This is a narrative paragraph, not a diagnosis name — skip
            pass
        else:
            # Try numbered patterns first: "(1) XXX" or "1. XXX 2. YYY"
            # Parenthesized: (1) XXX (2) YYY
            numbered_paren = re.findall(r'[（(]\s*\d+\s*[）)]\s*([^\n，。；（）]+)', raw)
            # Dotted: 1. XXX 2. YYY
            numbered_dot = re.findall(r'\d+[.、]\s*([^\n，。；\d]+)', raw)

            if numbered_paren:
                for n in numbered_paren:
                    n = _clean_single_diagnosis(n)
                    if n:
                        diags.append(n)
            elif numbered_dot:
                for n in numbered_dot:
                    n = _clean_single_diagnosis(n)
                    if n:
                        diags.append(n)
            else:
                # Single diagnosis — take up to first separator
                single = re.split(r'[。\n；]', raw)[0].strip()
                # Handle space-separated dual diagnoses:
                # "原发性醛固酮增多症     右侧肾上腺腺瘤"
                if re.search(r'\s{3,}', single):
                    parts = re.split(r'\s{3,}', single)
                    for p in parts:
                        p = _clean_single_diagnosis(p)
                        if p:
                            diags.append(p)
                else:
                    single = _clean_single_diagnosis(single)
                    if single:
                        diags.append(single)

    # Strategy 2: "诊断为XXX" or "诊断为：XXX"
    if not diags:
        m = re.search(r'诊断为[：:?\s]*([^\n，。；]+)', answer)
        if m:
            d = _clean_single_diagnosis(m.group(1).strip())
            if d:
                diags.append(d)

    # Strategy 3: "最可能的诊断是XXX"
    if not diags:
        m = re.search(r'(?:最可能|初步|首先考虑|考虑的)诊断[是为][：:?\s]*([^\n，。；]+)', answer)
        if m:
            d = _clean_single_diagnosis(m.group(1).strip())
            if d:
                diags.append(d)

    # Strategy 4: fallback — extract from title
    # title format: "XXX 案例分析-YYY" or "案例分析-YYY"
    if not diags and title:
        m = re.search(r'案例分析[-—]\s*(.+)', title)
        if m:
            d = m.group(1).strip()
            # Strip suffixes like "的诊疗", "的诊断与治疗"
            d = re.sub(r'(?:的诊疗|的诊断与治疗|的诊断及治疗|的诊断和鉴别诊断)$', '', d)
            d = _clean_single_diagnosis(d)
            if d:
                diags.append(d)

    # Deduplicate
    seen = set()
    result = []
    for d in diags:
        d_clean = d.strip()
        if d_clean and d_clean not in seen:
            seen.add(d_clean)
            result.append(d_clean)

    # Split compound diagnoses: "XXX导致的YYY" → ["XXX", "YYY"]
    split_result = []
    for d in result:
        m = re.match(r'(.+?)导致[的]?(.+)', d)
        if m:
            cause = m.group(1).strip()
            effect = m.group(2).strip()
            if len(cause) >= 2 and len(effect) >= 2:
                split_result.extend([cause, effect])
            else:
                split_result.append(d)
        else:
            split_result.append(d)

    # Deduplicate again after splitting
    seen2 = set()
    final = []
    for d in split_result:
        if d and d not in seen2:
            seen2.add(d)
            final.append(d)

    return final


def _clean_single_diagnosis(text: str) -> str | None:
    """清洗单条诊断名称：去除编号、解释性文本、碎片"""
    # Strip leading numbering: "1." or "(1)" or "①"
    text = re.sub(r'^\s*\d+[.、]\s*', '', text)
    text = re.sub(r'^\s*[（(]\s*\d+\s*[）)]\s*', '', text)

    # Strip severity/grading info in parentheses:
    # "勃起功能障碍（糖尿病性ED），中度（IIEF-5评分：11分）"
    # → "勃起功能障碍（糖尿病性ED）"
    # Keep etiology/type qualifiers like (糖尿病性ED) but remove severity like (中度)
    text = re.sub(r'[，,]\s*(?:轻度|中度|重度|高危|低危)[（(].*?[）)]', '', text)
    # Also remove standalone severity/grading parentheses at end
    text = re.sub(r'[（(]\s*(?:IIEF|评分|分级|分期|Grade|Stage).*?[）)]', '', text)

    # Strip explanatory text after colon within diagnosis name:
    # "海绵状血管瘤：是年轻人..." → "海绵状血管瘤"
    if '：' in text or ':' in text:
        parts = re.split(r'[：:]', text, maxsplit=1)
        text = parts[0].strip()

    # Strip trailing punctuation
    text = text.strip('，。；、')

    # Reject: too short, starts with narrative keywords, or is a whole sentence
    if len(text) < 2:
        return None
    if re.match(r'(?:病史|体征|检查|依据|分析|病人|本例|该)', text):
        return None
    if re.match(r'^[①②③④⑤⑥⑦⑧⑨⑩]', text):
        return None
    # Reject if looks like a sentence (has 3+ clause separators and >40 chars)
    if len(text) > 40 and len(re.findall(r'[，。；]', text)) >= 3:
        return None

    return text


def build_query(description: str) -> str:
    """从 description 拼成患者描述（作为 eval query）"""
    # description 包含现病史、体格检查等，直接用作 query
    # 截断到合理长度（~800字，避免 prompt 过长）
    text = description.strip()
    if len(text) > 800:
        text = text[:800] + "…"
    return text


def infer_difficulty(qa_question: str, description: str) -> str:
    """推断难度：鉴别诊断=hard，单诊断=medium，事实=easy"""
    # 鉴别诊断 explicitly in question → hard
    if '鉴别' in qa_question:
        return 'hard'

    # "诊断及诊断依据" is standard single diagnosis → medium
    if '诊断及诊断依据' in qa_question or '诊断与治疗' in qa_question:
        return 'medium'

    # "诊断" alone → medium (it asks for a diagnosis, not just facts)
    if '诊断' in qa_question:
        return 'medium'

    # Fallback: should not reach here (we only include 诊断/鉴别 questions)
    return 'easy'


def infer_min_hops(difficulty: str) -> int:
    """推断所需推理跳数"""
    if difficulty == 'hard':
        return 2  # 多症状 → PPR + 鉴别
    elif difficulty == 'medium':
        return 2
    else:
        return 1


def main():
    input_path = ROOT / "data" / "CMB-Clin-qa.json"
    output_path = ROOT / "data" / "eval_cmb_clin.jsonl"

    with open(input_path, encoding='utf-8') as f:
        data = json.load(f)

    eval_items = []
    extracted_count = 0
    total_qa = 0

    for item in data:
        description = build_query(item.get('description', ''))

        for qa in item['QA_pairs']:
            question = qa['question']
            total_qa += 1

            # Only include questions that ask for diagnosis or 鉴别
            # Skip 治疗原则/检查/手术 等非诊断问题
            if not ('诊断' in question or '鉴别' in question):
                continue

            # Skip multiple-choice questions (A.B.C.D format)
            if re.search(r'[A-E]\.\s*\S', question):
                continue

            gold = extract_diagnoses(qa['answer'], item.get('title', ''))
            if not gold:
                # Can't extract diagnosis — skip this item
                continue

            difficulty = infer_difficulty(question, description)
            min_hops = infer_min_hops(difficulty)

            eval_item = {
                "item_id": f"cmb_{item['id']}_{len(eval_items)}",
                "query": description,
                "gold_answer": None,
                "gold_diagnoses": gold,
                "min_hops_required": min_hops,
                "difficulty": difficulty,
                "user_id": "eval_user",
                # 保留原始字段供参考
                "_cmb_title": item.get('title', ''),
                "_cmb_question": question,
            }
            eval_items.append(eval_item)
            extracted_count += 1

    # Write JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in eval_items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # Stats
    difficulty_dist = {}
    for item in eval_items:
        d = item['difficulty']
        difficulty_dist[d] = difficulty_dist.get(d, 0) + 1

    print(f"CMB-Clin 原始数据: {len(data)} cases, {total_qa} QA pairs")
    print(f"适配后 EvalItem: {extracted_count} 条（含有效 gold_diagnoses）")
    print(f"难度分布: {difficulty_dist}")
    print(f"输出文件: {output_path}")


if __name__ == '__main__':
    main()