import argparse
import inspect
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from model.model_lora import apply_lora, load_lora
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


TCM_KEYWORDS = [
    "辨证", "脾", "胃", "肝", "肾", "气血", "阴虚", "阳虚", "湿", "痰湿",
    "风寒", "风热", "舌", "脉", "方剂", "中药", "调理", "经络", "寒热", "虚实",
]
WESTERN_KEYWORDS = [
    "检查", "诊断", "治疗", "药物", "抗生素", "止痛药", "退烧药", "感染", "炎症",
    "血压", "血糖", "血脂", "化验", "影像", "医生", "医院", "就医", "风险", "症状", "病因",
]

SYSTEM_PROMPTS = {
    "base": "你是一个谨慎、专业的医学健康问答助手。回答应清晰，并提醒用户必要时及时就医。",
    "tcm_lora": (
        "你是医学健康问答助手，当前使用中医 LoRA。请优先从中医辨证、调理原则、方剂或中药方向回答；"
        "不要给具体剂量，不要鼓励自行抓药；必要时提醒线下就医。"
    ),
    "western_lora": (
        "你是医学健康问答助手，当前使用西医 LoRA。请优先从现代医学病因、检查、治疗方案、用药安全和就医建议回答；"
        "不要把中药方剂作为主要方案；必要时提醒线下就医。"
    ),
}


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_model(base_weight, lora_path=None, device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "model")
    config = MiniMindConfig(hidden_size=768, num_hidden_layers=8, max_seq_len=8192, use_moe=False)
    model = MiniMindForCausalLM(config)
    model.load_state_dict(torch.load(base_weight, map_location=device), strict=True)
    if lora_path:
        apply_lora(model)
        load_lora(model, str(lora_path))
    model = model.half().eval().to(device)
    return model, tokenizer


def generate_answer(model, tokenizer, question, system_prompt, args):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        open_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(args.device)
    kwargs = {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": True,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": args.repetition_penalty,
    }
    if "no_repeat_ngram_size" in inspect.signature(model.generate).parameters:
        kwargs["no_repeat_ngram_size"] = args.no_repeat_ngram_size
    with torch.no_grad():
        output_ids = model.generate(**kwargs)
    answer_ids = output_ids[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(answer_ids, skip_special_tokens=True).strip()


def char_tokens(text):
    return [ch for ch in text if not ch.isspace()]


def distinct_n(text, n):
    tokens = char_tokens(text)
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(grams)) / max(len(grams), 1)


def max_ngram_repeat(text, n):
    tokens = char_tokens(text)
    if len(tokens) < n:
        return 0, ""
    grams = ["".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    if not grams:
        return 0, ""
    gram, count = Counter(grams).most_common(1)[0]
    return count, gram


def keyword_coverage(text, keywords):
    hits = [kw for kw in keywords if kw in text]
    return len(hits) / len(keywords), hits


def answer_metrics(answer):
    answer = answer or ""
    bigram_count, bigram = max_ngram_repeat(answer, 2)
    trigram_count, trigram = max_ngram_repeat(answer, 3)
    tcm_cov, tcm_hits = keyword_coverage(answer, TCM_KEYWORDS)
    western_cov, western_hits = keyword_coverage(answer, WESTERN_KEYWORDS)
    phrase_loop = bool(re.search(r"(.{1,6})(?:[，,。；;、\s]*\1){3,}", answer))
    suspected_repeat = bigram_count >= 12 or trigram_count >= 8 or phrase_loop
    return {
        "answered": len(answer.strip()) >= 8,
        "length": len(answer),
        "distinct_1": round(distinct_n(answer, 1), 4),
        "distinct_2": round(distinct_n(answer, 2), 4),
        "max_bigram_repeat": bigram_count,
        "max_bigram": bigram,
        "max_trigram_repeat": trigram_count,
        "max_trigram": trigram,
        "phrase_loop": phrase_loop,
        "suspected_repeat": suspected_repeat,
        "tcm_keyword_coverage": round(tcm_cov, 4),
        "tcm_keyword_hits": tcm_hits,
        "western_keyword_coverage": round(western_cov, 4),
        "western_keyword_hits": western_hits,
    }


def evaluate_model(model_key, model, tokenizer, prompts, args):
    rows = []
    for idx, item in enumerate(prompts, start=1):
        print(f"[{model_key}] {idx}/{len(prompts)} {item['id']}")
        answer = generate_answer(model, tokenizer, item["question"], SYSTEM_PROMPTS[model_key], args)
        row = {
            "id": item["id"],
            "category": item["category"],
            "question": item["question"],
            "model": model_key,
            "answer": answer,
            "metrics": answer_metrics(answer),
        }
        rows.append(row)
    return rows


def markdown_table(rows):
    lines = [
        "# 微调前后效果对比",
        "",
        "| ID | 类别 | 模型 | 是否回答 | 疑似复读 | 长度 | Distinct-1 | Distinct-2 | 中医覆盖 | 西医覆盖 | 回答摘要 |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        m = row["metrics"]
        answer = row["answer"].replace("\n", " ").replace("|", "/")
        summary = answer[:90] + ("..." if len(answer) > 90 else "")
        lines.append(
            f"| {row['id']} | {row['category']} | {row['model']} | "
            f"{'是' if m['answered'] else '否'} | {'是' if m['suspected_repeat'] else '否'} | "
            f"{m['length']} | {m['distinct_1']} | {m['distinct_2']} | "
            f"{m['tcm_keyword_coverage']} | {m['western_keyword_coverage']} | {summary} |"
        )
    lines.append("")
    lines.append("## 指标说明")
    lines.append("")
    lines.append("- 是否回答：回答长度至少 8 个字符。")
    lines.append("- 疑似复读：同一 2-gram 出现不少于 12 次，或同一 3-gram 出现不少于 8 次，或检测到短语循环。")
    lines.append("- distinct-1 / distinct-2：不同字符 unigram/bigram 占比，越低越可能重复。")
    lines.append("- 中医/西医覆盖：回答中命中的领域关键词比例，用于粗略观察风格迁移。")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Evaluate base vs TCM LoRA vs Western LoRA.")
    parser.add_argument("--eval_file", default=str(ROOT / "dataset/eval/medical_eval_prompts.jsonl"))
    parser.add_argument("--base_weight", default=str(ROOT / "out/full_sft_768.pth"))
    parser.add_argument("--tcm_lora", default=str(ROOT / "out/lora_medical_mix_768.pth"))
    parser.add_argument("--western_lora", default=str(ROOT / "out/lora_medical_768.pth"))
    parser.add_argument("--jsonl_output", default=str(ROOT / "eval_outputs/before_after_results.jsonl"))
    parser.add_argument("--md_output", default=str(ROOT / "eval_outputs/before_after_results.md"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.75)
    parser.add_argument("--repetition_penalty", type=float, default=1.25)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="Debug only: evaluate first N prompts.")
    args = parser.parse_args()

    prompts = list(read_jsonl(Path(args.eval_file)))
    if args.limit:
        prompts = prompts[:args.limit]

    all_rows = []
    model_specs = [
        ("base", None),
        ("tcm_lora", Path(args.tcm_lora)),
        ("western_lora", Path(args.western_lora)),
    ]
    for key, lora_path in model_specs:
        model, tokenizer = load_model(Path(args.base_weight), lora_path=lora_path, device=args.device)
        all_rows.extend(evaluate_model(key, model, tokenizer, prompts, args))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_jsonl(Path(args.jsonl_output), all_rows)
    md_path = Path(args.md_output)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(markdown_table(all_rows), encoding="utf-8")
    print(f"Saved JSONL: {args.jsonl_output}")
    print(f"Saved Markdown: {args.md_output}")


if __name__ == "__main__":
    main()
