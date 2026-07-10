import gradio as gr
import torch
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

REPO = "JustScriptzz/nexus-smAll-v1"

def load_model():
    import sys
    sys.path.insert(0, REPO)
    from src.model import Nexus
    from src.config import NexusConfig

    device = torch.device("cpu")
    config = NexusConfig()
    model = Nexus(config)
    weights_path = hf_hub_download(repo_id=REPO, filename="weights/nexus_instruct.pt")
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    model.eval()
    tokenizer_path = hf_hub_download(repo_id=REPO, filename="data/tokenizer.json")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    return model, tokenizer, config

model, tokenizer, config = load_model()
bos_id = tokenizer.token_to_id("<bos>") or 1
eos_id = tokenizer.token_to_id("<eos>") or 2

def chat(message, history):
    messages = history + [{"role": "user", "content": message}]
    tokens = [bos_id]
    for msg in messages:
        if msg["role"] == "user":
            tokens.extend(tokenizer.encode(f"User: {msg['content']}\nAssistant:").ids)
        elif msg["role"] == "assistant":
            tokens.extend(tokenizer.encode(f" {msg['content']}").ids + [eos_id])
    input_tensor = torch.tensor([tokens[-config.max_seq_len:]], dtype=torch.long)
    with torch.no_grad():
        for _ in range(128):
            seq_len = input_tensor.shape[1]
            if seq_len > config.max_seq_len:
                input_tensor = input_tensor[:, -config.max_seq_len:]
            logits = model(input_tensor, 0)
            logits = logits[:, -1, :] / 0.2
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_tensor = torch.cat([input_tensor, next_token], dim=-1)
            if next_token.item() == eos_id:
                break
    new_ids = input_tensor[0].tolist()[len(tokens):]
    reply = tokenizer.decode(new_ids)
    for tok in ["<|assistant|>", "<|user|>", "<|system|>"]:
        reply = reply.replace(tok, "")
    reply = reply.split("<eos>")[0].split("User:")[0].replace("Assistant:", "").strip()
    return reply or "..."

demo = gr.ChatInterface(
    fn=chat,
    title="Nexus SmAll v1",
    description="89.8M parameter transformer built from scratch",
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
