# 基于 MiniMind 的中西医 LoRA 微调系统

一个面向医学问答场景的 MiniMind LoRA 微调项目，支持 **基础模型 / 中医 LoRA / 西医 LoRA** 三种模式的推理、网页演示和微调前后效果评估。

本项目的重点不是“只训练出一个 LoRA 权重”，而是提供一套可以被复现、展示和分析的完整流程：从数据构建、LoRA 微调、网页问答，到固定测试集上的 **微调前 vs 微调后** 对比。

> 医疗免责声明：本项目仅用于自然语言处理、模型微调和教学展示。模型输出不能替代医生诊断、处方或治疗建议。

## 项目亮点

- 基于 MiniMind 小模型，适合个人 GPU 复现和课程/毕业设计展示。
- 分别训练并保留中医 LoRA 与西医 LoRA。
- 网页端支持一键切换：
  - 基础模型
  - 中医 LoRA
  - 西医 LoRA
- 提供固定评估集，包含：
  - 20 个中医问题
  - 20 个西医问题
  - 10 个通用问题
- 自动生成微调前后对比结果：
  - `base model`
  - `TCM LoRA`
  - `Western LoRA`
- 内置复读检测与简单量化指标：
  - 回答长度
  - distinct-1
  - distinct-2
  - 中医关键词覆盖率
  - 西医关键词覆盖率
  - 2-gram / 3-gram 重复检测

## 效果展示思路

本项目建议用同一个问题分别对比三类输出：

| 模型模式 | 预期表现 |
|---|---|
| 基础模型 | 通用健康问答，医学风格不一定稳定 |
| 中医 LoRA | 更偏向辨证、方剂、中药、调理建议 |
| 西医 LoRA | 更偏向病因、检查、治疗方案、用药安全、就医建议 |

例如对同一个问题：

```text
我感觉肚子有点不舒服，应该吃什么药呢？
```

可以观察：

- 中医 LoRA 是否更容易给出“辨证、脾胃、寒热虚实、调理建议”等表达。
- 西医 LoRA 是否更容易给出“可能病因、危险信号、检查、就医建议、不要自行用药”等表达。
- 是否出现重复输出，例如“当归 当归 当归……”或短语循环。

## 项目结构

```text
XinlinMedicalAgent
├── README.md
├── requirements.txt
├── configs
│   └── default_eval.json
├── dataset
│   ├── raw
│   ├── processed
│   │   ├── lora_medical_TCM.jsonl
│   │   ├── lora_medical_huatuo_clean.jsonl
│   │   ├── lora_medical_mix.jsonl
│   │   ├── lora_medical.jsonl
│   │   ├── train_tcm_lora.jsonl
│   │   └── train_western_lora.jsonl
│   ├── eval
│   │   └── medical_eval_prompts.jsonl
│   └── lm_dataset.py
├── scripts
│   ├── build_dataset.py
│   ├── build_lora_mix_dataset.py
│   ├── train_lora_tcm.py
│   ├── train_lora_western.py
│   ├── eval_before_after.py
│   ├── eval_lora_medical_prompts.py
│   └── web_medical_demo.py
├── model
│   ├── model_minimind.py
│   ├── model_lora.py
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── trainer
│   └── train_lora.py
├── out
│   ├── full_sft_768.pth
│   ├── lora_medical_mix_768.pth
│   └── lora_medical_768.pth
├── eval_outputs
└── docs
    ├── dataset_report.md
    └── minilora_analysis.md
```

## 环境准备

推荐环境：

- Python 3.10+
- NVIDIA GPU
- CUDA 版 PyTorch

安装依赖：

```bash
pip install -r requirements.txt
```

如果环境中没有 PyTorch，需要根据自己的 CUDA 版本单独安装。例如：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## 权重文件说明

本项目使用三个关键权重：

| 文件 | 说明 |
|---|---|
| `out/full_sft_768.pth` | 基础 SFT 模型权重 |
| `out/lora_medical_mix_768.pth` | 中医 LoRA 权重 |
| `out/lora_medical_768.pth` | 西医 LoRA 权重 |

如果你从 GitHub 克隆项目后没有这些大文件，需要自行放入 `out/` 目录，或根据训练脚本重新训练生成。

## 数据集说明

### 中医 LoRA 数据

主要来自：

- `dataset/processed/lora_medical_TCM.jsonl`
- `dataset/processed/lora_medical_huatuo_clean.jsonl`
- 少量通用 SFT 数据混合

最终混合数据：

```text
dataset/processed/lora_medical_mix.jsonl
```

用途：训练中医 LoRA，使模型更偏向中医辨证、方剂、中药和调理建议。

### 西医 LoRA 数据

主要文件：

```text
dataset/processed/lora_medical.jsonl
```

用途：训练西医 LoRA，使模型更偏向疾病科普、检查、治疗方案、用药安全和就医建议。

### 固定评估集

```text
dataset/eval/medical_eval_prompts.jsonl
```

包含 50 条固定问题：

- 20 条中医问题
- 20 条西医问题
- 10 条通用问题

固定评估集的作用是避免“只挑几个效果好的例子”，让微调效果更容易被客观展示。

## 数据处理

重新生成训练数据别名和数据统计报告：

```bash
python scripts/build_dataset.py
```

输出：

```text
dataset/processed/train_tcm_lora.jsonl
dataset/processed/train_western_lora.jsonl
docs/dataset_report.md
```

如果需要重新构建中医混合数据：

```bash
python scripts/build_lora_mix_dataset.py
```

## LoRA 微调

本项目使用 MiniMind 原生 LoRA 实现：

```text
model/model_lora.py
```

核心函数：

- `apply_lora(model)`
- `load_lora(model, path)`
- `save_lora(model, path)`

训练入口复用：

```text
trainer/train_lora.py
```

推荐训练参数：

| 参数 | 推荐值 |
|---|---|
| epochs | 2 |
| learning_rate | 1e-5 |
| batch_size | 32 |
| max_seq_len | 512 |

### 训练中医 LoRA

```bash
python scripts/train_lora_tcm.py
```

等价命令：

```bash
cd trainer
python train_lora.py ^
  --data_path ../dataset/processed/lora_medical_mix.jsonl ^
  --lora_name lora_medical_mix ^
  --epochs 2 ^
  --learning_rate 1e-5 ^
  --batch_size 32 ^
  --max_seq_len 512 ^
  --num_workers 0
```

### 训练西医 LoRA

```bash
python scripts/train_lora_western.py
```

等价命令：

```bash
cd trainer
python train_lora.py ^
  --data_path ../dataset/processed/lora_medical.jsonl ^
  --lora_name lora_medical ^
  --epochs 2 ^
  --learning_rate 1e-5 ^
  --batch_size 32 ^
  --max_seq_len 512 ^
  --num_workers 0
```

> Windows 环境建议保留 `--num_workers 0`，避免 DataLoader 多进程问题。

## 网页 Demo

启动网页端：

```bash
python -m streamlit run scripts/web_medical_demo.py --server.port 8502
```

浏览器打开：

```text
http://localhost:8502
```

网页功能：

- 选择基础模型 / 中医 LoRA / 西医 LoRA。
- 展示当前权重路径。
- 自动使用医疗安全提示词。
- 显示医疗免责声明。
- 切换模式时清空历史，避免不同模式的回答风格互相影响。

默认推理参数：

| 参数 | 默认值 |
|---|---|
| temperature | 0.6 |
| top_p | 0.75 |
| max_new_tokens | 512 |
| repetition_penalty | 1.25 |

当前 MiniMind 的 `generate()` 已支持 `repetition_penalty`。`no_repeat_ngram_size` 尚未在原生 `generate()` 中实现，代码会检测是否支持，不会强行传入导致报错。

## 微调前后评估

运行完整评估：

```bash
python scripts/eval_before_after.py
```

输出：

```text
eval_outputs/before_after_results.md
eval_outputs/before_after_results.jsonl
```

快速 smoke test：

```bash
python scripts/eval_before_after.py --limit 2 --jsonl_output eval_outputs/smoke_before_after.jsonl --md_output eval_outputs/smoke_before_after.md
```

评估脚本会对每个问题分别调用：

1. 基础模型
2. 中医 LoRA
3. 西医 LoRA

然后保存三组回答和指标。

## 评估指标

| 指标 | 含义 |
|---|---|
| answered | 回答长度是否足够，粗略判断是否有效回答 |
| length | 回答字符数 |
| distinct-1 | 不同字符 unigram 占比 |
| distinct-2 | 不同字符 bigram 占比 |
| tcm_keyword_coverage | 中医关键词覆盖率 |
| western_keyword_coverage | 西医关键词覆盖率 |
| suspected_repeat | 是否疑似复读 |

复读检测规则：

- 同一个 2-gram 出现不少于 12 次；或
- 同一个 3-gram 出现不少于 8 次；或
- 检测到短语循环。

这个评估不是医学正确性评估，而是用于展示：

- 微调后风格是否更贴近目标领域。
- 是否出现重复输出。
- 中医 / 西医 LoRA 是否体现出不同回答倾向。

## 参考项目 MiniLoRA 的迁移点

本项目参考了 MiniLoRA 的几个设计：

- 将数据处理、训练、推理对比、批量评估拆成独立脚本。
- 使用固定评估集，而不是临时提问。
- 保存 base vs LoRA 的结构化 JSONL 结果。
- 将结果汇总为 Markdown 表格，便于写报告和答辩展示。
- 保留数据统计报告，说明训练数据来源和规模。

详细分析见：

```text
docs/minilora_analysis.md
```

## 常见问题

### 1. 为什么中医和西医 LoRA 的回答有时差异不明显？

可能原因：

- 基础模型本身已经具备一定医学泛化能力。
- 中医和西医训练数据都属于医疗问答，语言风格有重叠。
- 问题本身过于宽泛，无法强制触发领域差异。
- 历史对话影响当前回答。

建议使用固定评估集中的同一问题对比三路输出，并查看关键词覆盖率和回答内容。

### 2. 为什么会出现复读？

可能原因：

- 训练数据中存在模板化回答。
- LoRA 数据量较少或风格过窄。
- 推理时 `max_new_tokens` 太长。
- 温度和 top_p 过高。
- 重复惩罚不足。

本项目默认使用较稳的推理参数：

```text
temperature=0.6
top_p=0.75
repetition_penalty=1.25
```

### 3. GitHub 上是否应该上传权重？

不建议直接上传大权重文件到普通 Git 仓库。推荐方式：

- 使用 Git LFS。
- 或在 README 中说明权重下载地址。
- 或只上传代码和小型评估文件，权重由用户自行训练生成。

### 4. 显存不够怎么办？

可以降低：

- `batch_size`
- `max_seq_len`
- `max_new_tokens`

也可以使用更小 batch 并配合梯度累积。

## 许可证与声明

请根据 MiniMind 原项目许可证和数据集许可证使用本项目。本项目仅用于学习、研究和工程展示，不提供真实医疗诊断能力。

如有急症、症状持续加重、儿童/孕产妇/老人等高风险情况，请及时到正规医疗机构就诊。
