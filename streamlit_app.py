import streamlit as st
import torch
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "api"))

from src.model import Nexus
from src.config import NexusConfig
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download

REPO = "JustScriptzz/nexus-smAll-v1"

st.set_page_config(
    page_title="Nexus SmAll v1",
    page_icon="🧠",
    layout="centered",
)

st.markdown("""
<style>
.stApp { background-color: #0f1117; color: #e1e1e1; }
.stChatInput textarea { background-color: #1e1e2e !important; color: #fff !important; border-color: #3a3a5c !important; }
.stChatMessage { background-color: #1a1a2e !important; border: 1px solid #2a2a4a !important; }
[data-testid="stChatMessageContent"] p { color: #e1e1e1 !important; }
.user-avatar { background-color: #7c3aed; }
.assistant-avatar { background-color: #1e1e3a; }
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.title("Nexus SmAll v1")
st.caption("89.8M parameter transformer — built from scratch")

@st.cache_resource(show_spinner="Loading Nexus SmAll v1...")
def load_model():
    device = torch.device("cpu")
    config = NexusConfig()
    model = Nexus(config)

    weights_path = hf_hub_download(repo_id=REPO, filename="weights/nexus_instruct.pt")
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tokenizer_path = hf_hub_download(repo_id=REPO, filename="data/tokenizer.json")
    tokenizer = Tokenizer.from_file(tokenizer_path)

    return model, tokenizer, config, device

model, tokenizer, config, device = load_model()
bos_id = tokenizer.token_to_id("<bos>") or 1
eos_id = tokenizer.token_to_id("<eos>") or 2

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm Nexus SmAll v1, a tiny 89.8M model trained from scratch. I might not always make sense, but ask me anything!"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Type a message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    tokens = [bos_id]
    for msg in st.session_state.messages[:-1]:
        if msg["role"] == "user":
            tokens.extend(tokenizer.encode(f"User: {msg['content']}\nAssistant:").ids)
        elif msg["role"] == "assistant":
            tokens.extend(tokenizer.encode(f" {msg['content']}").ids + [eos_id])

    tokens.extend(tokenizer.encode(f"User: {prompt}\nAssistant:").ids)

    input_tensor = torch.tensor([tokens[-config.max_seq_len:]], dtype=torch.long, device=device)

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
    response = tokenizer.decode(new_ids)
    response = response.split("<eos>")[0].split("User:")[0].replace("Assistant:", "").strip()

    if len(response) < 2:
        response = "..."

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)
