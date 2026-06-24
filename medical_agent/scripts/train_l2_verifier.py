"""
scripts/train_l2_verifier.py — 蒸馏 L2 Verifier (1.5B)

流程:
    1. 用 7B Teacher 在 (query, evidence, answer) 三元组上打分
    2. 在打分上微调 1.5B Student (Qwen2.5-1.5B-Instruct)
    3. 评估 Student 与 Teacher 的一致率 (AUC)

使用:
    python scripts/train_l2_verifier.py \
        --teacher-endpoint http://localhost:8000/v1 \
        --teacher-model qwen2.5-7b-medical-lora \
        --student-base-model Qwen/Qwen2.5-1.5B-Instruct \
        --train-data ./data/l2_train.jsonl \
        --output-dir ./verifier-1.5b

实现状态: 🟡 训练流程骨架 / ⚪ 实际跑需 GPU
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


VERIFIER_PROMPT_TEMPLATE = """你是医疗回答校验员。判断回答的质量。

用户问题: {query}
检索到的证据 (top-3):
{evidence}

回答内容: {answer}

请按以下维度评分 (0-1):
1. faithfulness: 回答是否忠于证据，没有编造证据外的内容?
2. relevance: 回答是否切题，回应了用户问题?
3. factuality: 回答中的医学事实是否准确?

输出严格的 JSON:
{{"faithfulness": <0-1>, "relevance": <0-1>, "factuality": <0-1>, "issues": ["..."]}}
"""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-endpoint", required=True)
    p.add_argument("--teacher-model", default="qwen2.5-7b-medical-lora")
    p.add_argument("--student-base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--train-data", required=True, help="JSONL 文件，每行 {query, evidence, answer}")
    p.add_argument("--output-dir", default="./verifier-1.5b")
    p.add_argument("--num-epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    return p.parse_args()


# ============================================================
# Step 1: Teacher 标注数据
# ============================================================

def teacher_label_data(
    train_data_path: str,
    teacher_endpoint: str,
    teacher_model: str,
) -> List[Dict]:
    """
    用 7B Teacher 给训练数据打分
    
    输入: JSONL，每行 {"query": ..., "evidence": [...], "answer": ...}
    输出: 同结构，加上 teacher_score 字段
    """
    from openai import OpenAI
    client = OpenAI(base_url=teacher_endpoint, api_key="EMPTY")
    
    labeled = []
    with open(train_data_path) as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            
            evidence_text = "\n".join(
                f"- {json.dumps(e, ensure_ascii=False)[:200]}"
                for e in item.get("evidence", [])[:3]
            )
            prompt = VERIFIER_PROMPT_TEMPLATE.format(
                query=item["query"],
                evidence=evidence_text,
                answer=item["answer"][:1000],
            )
            
            response = client.chat.completions.create(
                model=teacher_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.1,
            )
            
            try:
                response_text = response.choices[0].message.content
                import re
                match = re.search(r"\{.*\}", response_text, re.DOTALL)
                if match:
                    scores = json.loads(match.group(0))
                    item["teacher_label"] = scores
                    labeled.append(item)
            except Exception as e:
                logger.warning(f"Failed on item {i}: {e}")
            
            if (i + 1) % 100 == 0:
                logger.info(f"Labeled {i + 1} items")
    
    return labeled


# ============================================================
# Step 2: 训练 Student
# ============================================================

def build_sft_dataset(labeled_data: List[Dict]) -> List[Dict]:
    """
    把 teacher label 转成 SFT 训练样本
    
    Output format: {"messages": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]}
    """
    samples = []
    for item in labeled_data:
        evidence_text = "\n".join(
            f"- {json.dumps(e, ensure_ascii=False)[:200]}"
            for e in item.get("evidence", [])[:3]
        )
        user_msg = VERIFIER_PROMPT_TEMPLATE.format(
            query=item["query"],
            evidence=evidence_text,
            answer=item["answer"][:1000],
        )
        assistant_msg = json.dumps(item["teacher_label"], ensure_ascii=False)
        
        samples.append({
            "messages": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ]
        })
    return samples


def train_student(
    sft_data: List[Dict],
    base_model: str,
    output_dir: str,
    **train_args
):
    """
    SFT 训练 1.5B Student
    
    用 transformers + peft (QLoRA) 训练
    """
    # 生产环境完整实现:
    # from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    # from peft import LoraConfig, get_peft_model
    # from trl import SFTTrainer
    # 
    # tokenizer = AutoTokenizer.from_pretrained(base_model)
    # model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16)
    # 
    # lora_config = LoraConfig(
    #     r=16, lora_alpha=32, lora_dropout=0.05,
    #     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    # )
    # model = get_peft_model(model, lora_config)
    # 
    # training_args = TrainingArguments(
    #     output_dir=output_dir,
    #     num_train_epochs=train_args.get("num_epochs", 3),
    #     per_device_train_batch_size=train_args.get("batch_size", 4),
    #     learning_rate=train_args.get("learning_rate", 5e-5),
    #     ...
    # )
    # 
    # trainer = SFTTrainer(
    #     model=model, args=training_args,
    #     train_dataset=sft_data,
    #     tokenizer=tokenizer,
    # )
    # trainer.train()
    # trainer.save_model(output_dir)
    raise NotImplementedError("在 GPU 环境实现 SFT 训练（transformers + peft + trl）")


# ============================================================
# Step 3: 评估 Student vs Teacher 一致率
# ============================================================

def evaluate_student(
    student_endpoint: str,
    student_model: str,
    test_data: List[Dict],
) -> Dict:
    """
    对每条测试数据，比较 Student 和 Teacher 的评分
    
    指标:
    - Pearson correlation
    - 各维度 (faith / relevance / factuality) 的 MAE
    - 二分类 AUC (pass/fail，阈值 0.7)
    """
    raise NotImplementedError("实现 Student 推理 + 与 Teacher 对比")


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Teacher 标注
    logger.info("Step 1: Teacher labeling")
    labeled_data = teacher_label_data(
        args.train_data, args.teacher_endpoint, args.teacher_model
    )
    with open(output_dir / "labeled.jsonl", "w") as f:
        for item in labeled_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"Labeled {len(labeled_data)} items")
    
    # Step 2: 训练
    logger.info("Step 2: SFT training")
    sft_data = build_sft_dataset(labeled_data)
    train_student(
        sft_data,
        base_model=args.student_base_model,
        output_dir=str(output_dir / "model"),
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    
    # Step 3: 评估 (需要分独立测试集)
    logger.info("Step 3: Evaluation")
    # metrics = evaluate_student(...)
    
    logger.info("✅ L2 Verifier training complete!")
    logger.info(f"   Output: {output_dir}/model")


if __name__ == "__main__":
    main()
