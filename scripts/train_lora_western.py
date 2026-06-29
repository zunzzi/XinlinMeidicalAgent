import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    cmd = [
        sys.executable,
        "train_lora.py",
        "--data_path", "../dataset/processed/lora_medical.jsonl",
        "--lora_name", "lora_medical",
        "--epochs", "2",
        "--learning_rate", "1e-5",
        "--batch_size", "32",
        "--max_seq_len", "512",
        "--num_workers", "0",
    ]
    raise SystemExit(subprocess.call(cmd, cwd=ROOT / "trainer"))


if __name__ == "__main__":
    main()
