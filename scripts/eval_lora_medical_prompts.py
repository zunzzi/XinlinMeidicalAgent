import argparse
from pathlib import Path

from openai import OpenAI


PROMPTS = [
    "长期失眠、多梦、心烦，舌尖偏红，中医一般如何辨证调理？",
    "脾胃虚弱、饭后腹胀、大便溏薄，日常饮食和中医调理应注意什么？",
    "咽喉干痒、轻微咳嗽、无发热，从中医角度可能有哪些原因？",
    "女性手脚冰凉、怕冷、月经量少，中医如何看待？",
    "湿气重、头身困重、舌苔厚腻，适合哪些生活方式调整？",
    "肝郁气滞常见表现有哪些？可以如何调理情志和作息？",
    "气血不足导致乏力、面色淡白时，中医调理思路是什么？",
    "胃火旺、口臭、牙龈肿痛时，中医通常如何处理？",
    "阴虚火旺和实火上炎有什么区别？",
    "反复口腔溃疡，中医可能从哪些方面辨证？",
    "老年人膝关节酸痛、遇冷加重，中医如何理解？",
    "小儿食积有哪些常见表现？家长可以注意什么？",
    "产后气血亏虚时，饮食调养有哪些原则？",
    "痰湿体质的人如何减少痰湿生成？",
    "便秘但口干、舌红少津，中医调理方向是什么？",
    "夏季容易乏力、胸闷、食欲差，中医可能如何辨证？",
    "长期熬夜后眼干、急躁、口苦，中医怎么看？",
    "感冒初期怕冷、流清鼻涕，如何区分风寒和风热？",
    "中药当归常见功效是什么？哪些情况不宜自行大量使用？",
    "请用安全、谨慎的方式说明中医调理为什么需要面诊辨证。",
]


def main():
    parser = argparse.ArgumentParser(description="Evaluate fixed medical LoRA prompts through OpenAI-compatible API.")
    parser.add_argument("--base_url", default="http://localhost:8998/v1")
    parser.add_argument("--api_key", default="sk-123")
    parser.add_argument("--model", default="minimind-local:latest")
    parser.add_argument("--output", default="../eval_outputs/lora_medical_mix_eval.txt")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.75)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--repetition_penalty", type=float, default=1.25)
    args = parser.parse_args()

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="\n") as f:
        for idx, prompt in enumerate(PROMPTS, start=1):
            response = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                extra_body={
                    "open_thinking": False,
                    "chat_template_kwargs": {"open_thinking": False},
                    "repetition_penalty": args.repetition_penalty,
                },
            )
            answer = response.choices[0].message.content or ""
            f.write(f"## {idx}. {prompt}\n\n{answer.strip()}\n\n")
            f.write("-" * 80 + "\n\n")
            print(f"[{idx:02d}/{len(PROMPTS)}] done")

    print(f"Saved evaluation output to {output_path}")


if __name__ == "__main__":
    main()
