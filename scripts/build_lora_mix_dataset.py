import argparse
import json
import random
from pathlib import Path


def normalize_sample(obj):
    conversations = obj.get("conversations") if isinstance(obj, dict) else None
    if not isinstance(conversations, list) or len(conversations) < 2:
        return None

    user_content = None
    assistant_content = None
    for message in conversations:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if not isinstance(content, str):
            continue
        content = content.strip()
        if not content:
            continue
        if role == "user" and user_content is None:
            user_content = content
        elif role == "assistant" and assistant_content is None:
            assistant_content = content
        if user_content and assistant_content:
            break

    if not user_content or not assistant_content:
        return None

    return {
        "conversations": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def sample_key(sample):
    conv = sample["conversations"]
    return json.dumps([conv[0]["content"], conv[1]["content"]], ensure_ascii=False, sort_keys=True)


def reservoir_load(path, sample_size, seed):
    stats = {
        "path": str(path),
        "loaded": 0,
        "invalid": 0,
        "duplicates": 0,
    }
    sampled = []
    seen = set()
    rng = random.Random(seed)

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                stats["invalid"] += 1
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                stats["invalid"] += 1
                continue

            sample = normalize_sample(obj)
            if sample is None:
                stats["invalid"] += 1
                continue

            key = sample_key(sample)
            if key in seen:
                stats["duplicates"] += 1
                continue
            seen.add(key)
            stats["loaded"] += 1

            if sample_size <= 0:
                continue
            if len(sampled) < sample_size:
                sampled.append(sample)
            else:
                j = rng.randint(0, stats["loaded"] - 1)
                if j < sample_size:
                    sampled[j] = sample

    return sampled, stats


def compute_counts(valid_counts, ratios):
    total = min(valid_counts[name] // ratio for name, ratio in ratios.items() if ratio > 0)
    counts = {name: int(total * ratio) for name, ratio in ratios.items()}
    remainder = total - sum(counts.values())
    for name, _ in sorted(ratios.items(), key=lambda item: item[1], reverse=True):
        if remainder <= 0:
            break
        counts[name] += 1
        remainder -= 1
    return total, counts


def count_valid(path):
    loaded = 0
    invalid = 0
    duplicates = 0
    seen = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                invalid += 1
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                invalid += 1
                continue
            sample = normalize_sample(obj)
            if sample is None:
                invalid += 1
                continue
            key = sample_key(sample)
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            loaded += 1
    return {"loaded": loaded, "invalid": invalid, "duplicates": duplicates}


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return (Path(__file__).resolve().parent / path).resolve()


def main():
    parser = argparse.ArgumentParser(description="Build mixed LoRA medical dataset.")
    parser.add_argument("--tcm", default="../dataset/lora_medical_TCM.jsonl")
    parser.add_argument("--huatuo", default="../dataset/lora_medical_huatuo_clean.jsonl")
    parser.add_argument("--sft", default="../dataset/sft_t2t.jsonl")
    parser.add_argument("--output", default="../dataset/lora_medical_mix.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sources = {
        "TCM": resolve_path(args.tcm),
        "Huatuo": resolve_path(args.huatuo),
    }
    ratios = {"TCM": 0.8, "Huatuo": 0.2}
    sft_path = resolve_path(args.sft)
    if sft_path.exists():
        sources["SFT"] = sft_path
        ratios = {"TCM": 0.7, "Huatuo": 0.2, "SFT": 0.1}

    for name, path in sources.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} source not found: {path}")

    print("Counting TCM valid samples to determine mix size...")
    tcm_stats = count_valid(sources["TCM"])
    target_total = tcm_stats["loaded"] // ratios["TCM"]
    target_counts = {name: int(target_total * ratio) for name, ratio in ratios.items()}
    target_counts["TCM"] = tcm_stats["loaded"]

    print("Sampling sources...")
    all_samples = []
    final_seen = set()
    sampled_stats = {}
    for index, (name, path) in enumerate(sources.items()):
        sampled, stats = reservoir_load(path, target_counts[name], args.seed + index)
        sampled_stats[name] = stats
        added = 0
        for sample in sampled:
            key = sample_key(sample)
            if key in final_seen:
                continue
            final_seen.add(key)
            all_samples.append(sample)
            added += 1
        sampled_stats[name]["sampled"] = len(sampled)
        sampled_stats[name]["added_after_global_dedup"] = added

    rng = random.Random(args.seed)
    rng.shuffle(all_samples)

    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print("\nMix ratios:")
    for name, ratio in ratios.items():
        print(f"  {name}: {ratio:.0%}")
    print("\nStatistics:")
    for name in sources:
        stat = sampled_stats[name]
        print(
            f"  {name}: loaded={stat['loaded']}, invalid={stat['invalid']}, "
            f"duplicates={stat['duplicates']}, sampled={stat['sampled']}, "
            f"saved={stat['added_after_global_dedup']}"
        )
    print(f"\nSaved: {len(all_samples)} samples -> {output_path}")


if __name__ == "__main__":
    main()
