import argparse
import json
import random
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TCM_KEYWORDS = ["辨证", "中药", "方剂", "气血", "脾胃", "阴虚", "阳虚", "湿", "舌", "脉"]
WESTERN_KEYWORDS = ["检查", "治疗", "诊断", "药物", "抗生素", "血压", "血糖", "医生", "医院", "感染"]


def normalize(obj):
    conversations = obj.get("conversations") if isinstance(obj, dict) else None
    if not isinstance(conversations, list):
        return None
    user = None
    assistant = None
    for msg in conversations:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        if role == "user" and user is None:
            user = content.strip()
        elif role == "assistant" and assistant is None:
            assistant = content.strip()
    if not user or not assistant:
        return None
    return {"conversations": [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]}


def read_jsonl(path):
    stats = {"raw": 0, "valid": 0, "invalid": 0, "duplicates": 0}
    rows = []
    seen = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stats["raw"] += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                stats["invalid"] += 1
                continue
            item = normalize(obj)
            if item is None:
                stats["invalid"] += 1
                continue
            key = json.dumps(item["conversations"], ensure_ascii=False, sort_keys=True)
            if key in seen:
                stats["duplicates"] += 1
                continue
            seen.add(key)
            stats["valid"] += 1
            rows.append(item)
    return rows, stats


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def keyword_counts(rows, keywords):
    counter = Counter()
    for row in rows:
        text = row["conversations"][0]["content"] + "\n" + row["conversations"][1]["content"]
        for kw in keywords:
            if kw in text:
                counter[kw] += 1
    return counter


def sample_rows(rows, n, rng):
    if n >= len(rows):
        return list(rows)
    return rng.sample(rows, n)


def build_report(path, sections):
    lines = ["# 数据集统计报告", ""]
    for name, info in sections.items():
        lines.append(f"## {name}")
        lines.append("")
        lines.append(f"- 原始行数：{info['stats']['raw']}")
        lines.append(f"- 有效样本：{info['stats']['valid']}")
        lines.append(f"- 无效样本：{info['stats']['invalid']}")
        lines.append(f"- 去重样本：{info['stats']['duplicates']}")
        lines.append(f"- 输出样本：{info['output_count']}")
        lines.append(f"- 输出文件：`{info['output']}`")
        lines.append("")
        lines.append("关键词命中：")
        for kw, count in info["keywords"].most_common():
            lines.append(f"- {kw}: {count}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Build and report medical LoRA datasets.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tcm_source", default=str(ROOT / "dataset/processed/lora_medical_mix.jsonl"))
    parser.add_argument("--western_source", default=str(ROOT / "dataset/processed/lora_medical.jsonl"))
    parser.add_argument("--tcm_output", default=str(ROOT / "dataset/processed/train_tcm_lora.jsonl"))
    parser.add_argument("--western_output", default=str(ROOT / "dataset/processed/train_western_lora.jsonl"))
    parser.add_argument("--max_tcm", type=int, default=0)
    parser.add_argument("--max_western", type=int, default=0)
    parser.add_argument("--report", default=str(ROOT / "docs/dataset_report.md"))
    args = parser.parse_args()

    rng = random.Random(args.seed)
    tcm_rows, tcm_stats = read_jsonl(Path(args.tcm_source))
    western_rows, western_stats = read_jsonl(Path(args.western_source))
    rng.shuffle(tcm_rows)
    rng.shuffle(western_rows)
    if args.max_tcm:
        tcm_rows = sample_rows(tcm_rows, args.max_tcm, rng)
    if args.max_western:
        western_rows = sample_rows(western_rows, args.max_western, rng)

    write_jsonl(Path(args.tcm_output), tcm_rows)
    write_jsonl(Path(args.western_output), western_rows)
    sections = {
        "中医 LoRA 数据": {
            "stats": tcm_stats,
            "output_count": len(tcm_rows),
            "output": args.tcm_output,
            "keywords": keyword_counts(tcm_rows, TCM_KEYWORDS),
        },
        "西医 LoRA 数据": {
            "stats": western_stats,
            "output_count": len(western_rows),
            "output": args.western_output,
            "keywords": keyword_counts(western_rows, WESTERN_KEYWORDS),
        },
    }
    build_report(Path(args.report), sections)
    print(f"Saved TCM dataset: {len(tcm_rows)} -> {args.tcm_output}")
    print(f"Saved Western dataset: {len(western_rows)} -> {args.western_output}")
    print(f"Saved report: {args.report}")


if __name__ == "__main__":
    main()
