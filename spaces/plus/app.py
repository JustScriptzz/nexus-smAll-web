import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import hf_hub_download

REPO = "JustScriptzz/nexus-plus-v2"

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        REPO,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    return model, tokenizer

model, tokenizer = load_model()

@torch.inference_mode()
def chat(message, history):
    messages = [{"role": "user", "content": message}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.8,
        do_sample=True,
    )
    reply = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return reply or "..."

demo = gr.ChatInterface(
    fn=chat,
    title="Nexus Plus v2",
    description="Qwen3-4B fine-tuned with QLoRA — smarter, deeper reasoning",
    theme=gr.themes.Base(
        primary_hue="purple",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ).set(
        body_background_fill="#0a0a0f",
        body_text_color="#e1e1e6",
        block_background_fill="#111118",
        block_border_color="#1e1e2e",
        block_label_text_color="#888",
        input_background_fill="#111118",
        input_background_fill_focus="#111118",
        button_primary_background_fill="#7c3aed",
        button_primary_background_fill_hover="#6d28d9",
    ),
)

demo.launch()
