# MiniLoRA 参考项目分析

参考路径：`D:\miniLora\MiniLoRA-master\MiniLoRA-master`

## 1. 数据集格式

MiniLoRA 使用中文医疗 SFT 数据，原始格式接近 Alpaca：

```json
{"instruction":"...","input":"...","output":"..."}
```

数据准备脚本会统一清洗为 `instruction/input/output` 三字段，并在训练时转换为 chat messages：

```json
[
  {"role":"system","content":"医疗健康问答助手提示词"},
  {"role":"user","content":"instruction + 问题"},
  {"role":"assistant","content":"output"}
]
```

训练阶段会对 prompt 部分 label 置为 `-100`，只让 assistant 回答部分参与 loss。

## 2. LoRA 微调入口脚本

主要入口：

- `scripts/train_lora.py`：完整参考实现。
- `scripts/my_train_lora.py`：教学版 TODO 模板。

## 3. 训练参数

`scripts/train_lora.py` 的核心默认参数：

- 模型：`models/Qwen2.5-0.5B-Instruct`
- `r=8`
- `lora_alpha=16`
- `lora_dropout=0.05`
- `target_modules`: `q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj`
- `epochs=1`
- `batch_size=1`
- `grad_accum=8`
- `lr=2e-4`
- `max_length=1024`
- `logging_steps=10`
- `save_steps=100`
- `eval_steps=100`

## 4. 基础模型和 LoRA 加载方式

MiniLoRA 使用 HuggingFace + PEFT：

- base：`AutoModelForCausalLM.from_pretrained(model_name)`
- LoRA：`PeftModel.from_pretrained(base_model, adapter_dir)`
- tokenizer 优先从 adapter 目录加载，否则从 base 模型目录加载。

## 5. 推理测试

主要文件：

- `scripts/infer_compare.py`：完整 base vs LoRA 单条问题对比。
- `scripts/my_infer_compare.py`：教学版 TODO 模板。

流程：

1. 加载 base 模型生成回答。
2. 释放显存。
3. 加载 base + LoRA 生成回答。
4. 打印两者回答和长度，人工观察专业性、条理性、医疗建议倾向。

## 6. 评估、对比、可视化和 README 示例

存在：

- `data/medical/eval_prompts.jsonl`：固定评估问题。
- `results/base_vs_lora.jsonl`：base vs LoRA 结果。
- `results/lora_ablation_summary.csv`：rank、数据量等消融结果。
- README 中给出模块化运行流程和结果观察点。

批量评估脚本方面，`my_eval_lora.py` 是教学 TODO 模板；项目已有结果文件用于展示。

## 7. 如何显式展示“微调前 vs 微调后”

MiniLoRA 的展示方式值得迁移：

- 固定评估问题集，避免临时挑问题。
- 同一问题分别生成 base 和 LoRA 回答。
- 保存结构化 JSONL 结果，便于复查。
- 额外保存表格/CSV 总结，如 loss、rank、训练样本量。
- README 中写出观察点：是否更专业、是否更有条理、是否更贴合医疗场景。

## 8. 值得迁移的设计

- `data/scripts/results/docs` 的清晰分层。
- 数据准备、训练、推理对比、批量评估拆成独立脚本。
- 固定 eval prompts。
- base vs LoRA 对比结果落盘。
- 指标表格和实验结论写入 README。
- smoke test 和完整评估分离，便于快速验证代码。
