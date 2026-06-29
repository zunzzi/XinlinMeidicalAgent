import inspect
import os
import random
import re
import sys
from threading import Thread

import numpy as np
import torch
import streamlit as st
from transformers import AutoTokenizer, TextIteratorStreamer

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.model_lora import apply_lora, load_lora
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


APP_TITLE = "杏林智诊助手"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
OUT_DIR = os.path.join(PROJECT_ROOT, "out")

BASE_WEIGHT = os.path.join(OUT_DIR, "full_sft_768.pth")
LORA_MODES = {
    "基础模型": {
        "label": "基础模型",
        "path": None,
        "hint": "不加载 LoRA，用于观察微调前回答。",
        "system": (
            "你是杏林智诊助手，当前处于【基础模型】模式。"
            "请以通用健康问答方式回答，保持谨慎，不要冒充医生。"
            "结尾提醒：医疗回答仅供参考，不能替代医生诊断。"
        ),
    },
    "中医模式": {
        "label": "中医",
        "path": os.path.join(OUT_DIR, "lora_medical_mix_768.pth"),
        "hint": "偏向中医辨证、方剂、中药、调理建议。",
        "system": (
            "你是杏林智诊agent，当前处于【中医模式】。"
            "必须优先使用中医表达体系回答：先说明可能的中医辨证方向，再给出调理原则、饮食起居建议，必要时列出常见方剂或中药方向。"
            "不要把抗生素、止痛药、利尿剂等西药作为主要方案；如果用户症状可能危险，只在安全提醒里建议及时线下就医。"
            "涉及方剂和中药时不要给具体剂量，不要鼓励自行抓药。"
            "结尾提醒：医疗回答仅供参考，不能替代医生诊断。"
        ),
    },
    "西医模式": {
        "label": "西医",
        "path": os.path.join(OUT_DIR, "lora_medical_768.pth"),
        "hint": "偏向疾病科普、检查、治疗方案、就医建议。",
        "system": (
            "你是杏林智诊agent，当前处于【西医模式】。"
            "必须优先使用现代医学表达体系回答：先说明可能原因或鉴别方向，再给出需要关注的症状、检查建议、常规处理和就医建议。"
            "不要推荐中药、方剂、辨证分型或中医调理作为主要方案；可以提醒不要自行用药，处方药需医生评估。"
            "如有急腹痛、发热、呕血黑便、持续加重等危险信号，要建议及时就医。"
            "结尾提醒：医疗回答仅供参考，不能替代医生诊断。"
        ),
    },
}

DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.75
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_REPETITION_PENALTY = 1.25

device = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(page_title=APP_TITLE, initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
        .stButton button {
            border-radius: 50% !important;
            width: 32px !important;
            height: 32px !important;
            padding: 0 !important;
            background-color: transparent !important;
            border: 1px solid #ddd !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 14px !important;
            color: #666 !important;
            margin: 5px 10px 5px 0 !important;
        }
        .stButton button:hover {
            border-color: #999 !important;
            color: #333 !important;
            background-color: #f5f5f5 !important;
        }
        .stMainBlockContainer > div:first-child {
            margin-top: -50px !important;
        }
        .stApp > div:last-child {
            margin-bottom: -35px !important;
        }
        .medical-mode {
            color: #bbb;
            font-size: 14px;
            line-height: 1.7;
            margin-top: 6px;
            margin-bottom: 12px;
        }
        .medical-disclaimer {
            color: #c7a66a;
            font-style: italic;
            margin-top: 2px;
            margin-bottom: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def process_assistant_content(content):
    if "<think>" in content and "</think>" in content:
        def format_think(match):
            think_content = match.group(2)
            if think_content.replace("\n", "").strip():
                return (
                    '<details open style="border-left: 2px solid #666; padding-left: 12px; margin: 8px 0;">'
                    '<summary style="cursor: pointer; color: #888;">已思考</summary>'
                    f'<div style="color: #aaa; font-size: 0.95em; margin-top: 8px; max-height: 100px; overflow-y: auto;">{think_content.strip()}</div>'
                    "</details>"
                )
            return ""
        content = re.sub(r"(<think>)(.*?)(</think>)", format_think, content, flags=re.DOTALL)

    if "<think>" in content and "</think>" not in content:
        def format_think_in_progress(match):
            think_content = match.group(1)
            return (
                '<details open style="border-left: 2px solid #666; padding-left: 12px; margin: 8px 0;">'
                '<summary style="cursor: pointer; color: #888;">思考中...</summary>'
                f'<div style="color: #aaa; font-size: 0.95em; margin-top: 8px; max-height: 100px; overflow-y: auto;">{think_content.strip().replace(chr(10), "<br>")}</div>'
                "</details>"
            )
        content = re.sub(r"<think>(.*?)$", format_think_in_progress, content, flags=re.DOTALL)

    return content


@st.cache_resource
def load_base_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    lm_config = MiniMindConfig(hidden_size=768, num_hidden_layers=8, max_seq_len=8192, use_moe=False)
    model = MiniMindForCausalLM(lm_config)
    model.load_state_dict(torch.load(BASE_WEIGHT, map_location=device), strict=True)
    apply_lora(model)
    model = model.half().eval().to(device)
    return model, tokenizer


def load_lora_for_mode(model, mode_name):
    mode = LORA_MODES[mode_name]
    if mode["path"] is None:
        current = st.session_state.get("current_lora_mode")
        if current != mode_name:
            # 基础模型模式通过加载一个全零临时 LoRA 状态避免残留上一次 LoRA 权重。
            # 当前项目的 LoRA 层 B 矩阵初始化为 0，因此重新 apply_lora 后的初始状态等价于 base。
            # Streamlit 已缓存 model；为了彻底清除 LoRA 权重，这里要求用户切到基础模型时重新加载页面资源。
            st.cache_resource.clear()
            st.session_state.current_lora_mode = mode_name
            st.rerun()
        return
    if not os.path.exists(mode["path"]):
        raise FileNotFoundError(f"LoRA 权重不存在: {mode['path']}")
    current = st.session_state.get("current_lora_mode")
    if current != mode_name:
        # 当前项目 LoRA 实现是在同一个 model 上覆盖各层 module.lora 权重。
        # 因此这里缓存 base model，不重复加载完整模型；切换模式时只重新加载 LoRA 权重。
        load_lora(model, mode["path"])
        st.session_state.current_lora_mode = mode_name


def clear_history():
    st.session_state.messages = []
    st.session_state.chat_messages = []


def render_history(messages):
    for message in messages:
        if message["role"] == "assistant":
            st.markdown(process_assistant_content(message["content"]), unsafe_allow_html=True)
        elif message["role"] == "user":
            st.markdown(
                f'<div style="display: flex; justify-content: flex-end;"><div style="display: inline-block; margin: 10px 0; padding: 8px 12px; background-color: #3d4450; border-radius: 22px; color: white;">{message["content"]}</div></div>',
                unsafe_allow_html=True,
            )


def build_generation_kwargs(model, inputs, tokenizer, streamer):
    kwargs = {
        "input_ids": inputs.input_ids,
        "max_new_tokens": st.session_state.max_new_tokens,
        "num_return_sequences": 1,
        "do_sample": True,
        "attention_mask": inputs.attention_mask,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "temperature": DEFAULT_TEMPERATURE,
        "top_p": DEFAULT_TOP_P,
        "streamer": streamer,
        "repetition_penalty": DEFAULT_REPETITION_PENALTY,
    }
    params = inspect.signature(model.generate).parameters
    if "no_repeat_ngram_size" in params:
        kwargs["no_repeat_ngram_size"] = 4
    else:
        # MiniMind 当前原生 generate 未实现 no_repeat_ngram_size；如需启用，需要到
        # model/model_minimind.py 的 MiniMindForCausalLM.generate 中增加该逻辑。
        pass
    return kwargs


selected_mode = st.sidebar.radio("问诊模式", list(LORA_MODES.keys()), index=0)
previous_mode = st.session_state.get("last_selected_mode")
if previous_mode and previous_mode != selected_mode:
    clear_history()
st.session_state.last_selected_mode = selected_mode
mode_info = LORA_MODES[selected_mode]

st.sidebar.markdown('<hr style="margin: 12px 0 16px 0;">', unsafe_allow_html=True)
st.sidebar.caption(f"当前模式：{mode_info['label']}")
st.sidebar.caption(mode_info["hint"])
st.sidebar.caption(f"当前权重：{mode_info['path'] or BASE_WEIGHT}")
st.sidebar.caption("医疗回答仅供参考，不能替代医生诊断。")

st.sidebar.markdown('<hr style="margin: 12px 0 16px 0;">', unsafe_allow_html=True)
st.session_state.history_chat_num = st.sidebar.slider("历史对话轮次", 0, 8, 0, step=2)
st.session_state.max_new_tokens = st.sidebar.slider("最大生成长度", 128, 1024, DEFAULT_MAX_NEW_TOKENS, step=32)

if st.sidebar.button("清空历史"):
    clear_history()
    st.rerun()

st.markdown(
    f'<div style="display: flex; flex-direction: column; align-items: center; text-align: center; margin: 0; padding: 0;">'
    '<div style="font-style: italic; font-weight: 900; margin: 0; padding-top: 4px; display: flex; align-items: center; justify-content: center; flex-wrap: wrap; width: 100%;">'
    f'<span style="font-size: 28px;">{APP_TITLE}</span>'
    "</div>"
    f'<div class="medical-mode">当前模式：{mode_info["label"]}｜{mode_info["hint"]}</div>'
    '<div class="medical-disclaimer">医疗回答仅供参考，不能替代医生诊断；如有急症或持续不适，请及时线下就医。</div>'
    "</div>",
    unsafe_allow_html=True,
)


def main():
    model, tokenizer = load_base_model()
    load_lora_for_mode(model, selected_mode)

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_messages = []

    messages = st.session_state.messages
    render_history(messages)

    prompt = st.chat_input(key="input", placeholder=f"向{APP_TITLE}描述症状或咨询问题")
    if not prompt:
        return

    st.markdown(
        f'<div style="display: flex; justify-content: flex-end;"><div style="display: inline-block; margin: 10px 0; padding: 8px 12px; background-color: #3d4450; border-radius: 22px; color: white;">{prompt}</div></div>',
        unsafe_allow_html=True,
    )

    messages.append({"role": "user", "content": prompt[-st.session_state.max_new_tokens:]})
    st.session_state.chat_messages.append({"role": "user", "content": prompt[-st.session_state.max_new_tokens:]})

    placeholder = st.empty()
    status_box = st.status(f"{mode_info['label']} LoRA 正在生成回答...", expanded=False)
    with status_box:
        setup_seed(random.randint(0, 2**32 - 1))
        system_prompt = [{"role": "system", "content": mode_info["system"]}]
        st.session_state.chat_messages = system_prompt + st.session_state.chat_messages[-(st.session_state.history_chat_num + 1):]
        new_prompt = tokenizer.apply_chat_template(
            st.session_state.chat_messages,
            tokenize=False,
            add_generation_prompt=True,
            open_thinking=False,
        )
        inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = build_generation_kwargs(model, inputs, tokenizer, streamer)
        Thread(target=model.generate, kwargs=generation_kwargs).start()

        answer = ""
        for new_text in streamer:
            answer += new_text
            placeholder.markdown(process_assistant_content(answer), unsafe_allow_html=True)
        status_box.update(label=f"{mode_info['label']} LoRA 回答完成", state="complete", expanded=False)

    messages.append({"role": "assistant", "content": answer})
    st.session_state.chat_messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
