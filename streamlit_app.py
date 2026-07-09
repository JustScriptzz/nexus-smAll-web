import streamlit as st
import torch
import time
import datetime

from huggingface_hub import hf_hub_download, InferenceClient

REPO_SMALL = "JustScriptzz/nexus-smAll-v1"
REPO_PLUS = "JustScriptzz/nexus-plus-v2"

st.set_page_config(
    page_title="Nexus AI",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

.stApp {
    background-color: #0a0a0f;
    color: #e1e1e6;
    font-family: 'Inter', sans-serif;
}

[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { display: none; }

div[data-testid="stChatMessage"] {
    border-radius: 16px;
    padding: 12px 16px;
    margin: 8px 0;
    border: 1px solid rgba(255,255,255,0.06);
}

div[data-testid="stChatMessage"][aria-label="user"] {
    background: linear-gradient(135deg, #7c3aed22, #6d28d922);
    border-color: #7c3aed33;
}

div[data-testid="stChatMessage"][aria-label="assistant"] {
    background: rgba(255,255,255,0.03);
}

[data-testid="stChatMessageContent"] p {
    color: #e1e1e6 !important;
    font-size: 15px;
    line-height: 1.6;
}

div[data-testid="stChatInput"] {
    background: transparent;
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
}

div[data-testid="stChatInput"] textarea {
    background: #111118 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 16px !important;
    font-size: 15px;
}

div[data-testid="stChatInput"] textarea:focus {
    box-shadow: 0 0 0 2px #7c3aed44;
}

footer { visibility: hidden; }
header { visibility: hidden; }

.typing-indicator {
    display: inline-flex;
    gap: 4px;
    padding: 8px 14px;
    background: rgba(255,255,255,0.04);
    border-radius: 12px;
    margin: 8px 0;
}
.typing-dot {
    width: 7px; height: 7px;
    background: #7c3aed;
    border-radius: 50%;
    animation: bounce 1.4s infinite;
}
.typing-dot:nth-child(2) { animation-delay: 0.16s; }
.typing-dot:nth-child(3) { animation-delay: 0.32s; }
@keyframes bounce {
    0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
    30% { transform: translateY(-6px); opacity: 1; }
}

.status-bar {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 20px;
    font-size: 12px;
    color: #888;
    width: fit-content;
    margin: 0 auto 12px;
}
.status-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #22c55e;
}
.status-dot.loading {
    background: #eab308;
    animation: pulse 1s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

.log-entry {
    padding: 4px 0;
    font-size: 12px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    line-height: 1.4;
}
.log-time { color: #555; }
.log-info { color: #a78bfa; }
.log-ok { color: #22c55e; }
.log-warn { color: #eab308; }
.log-err { color: #ef4444; }
.log-dim { color: #444; }
</style>
""", unsafe_allow_html=True)

if "logs" not in st.session_state:
    st.session_state.logs = []
if "log_id" not in st.session_state:
    st.session_state.log_id = 0

def log(msg, level="info"):
    st.session_state.log_id += 1
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append({
        "id": st.session_state.log_id,
        "time": ts,
        "msg": msg,
        "level": level,
    })
    if len(st.session_state.logs) > 100:
        st.session_state.logs = st.session_state.logs[-100:]

with st.sidebar:
    st.markdown("### Logs")
    auto_refresh = st.checkbox("Auto-refresh", value=True, key="auto_refresh_logs")
    if st.button("Clear logs"):
        st.session_state.logs = []
        st.rerun()
    st.markdown("---")

    if not st.session_state.logs:
        st.caption("No activity yet")
    else:
        html = ""
        for entry in reversed(st.session_state.logs):
            cls = f"log-{entry['level']}"
            html += f'<div class="log-entry"><span class="log-time">[{entry["time"]}]</span> <span class="{cls}">{entry["msg"]}</span></div>'
        st.markdown(html, unsafe_allow_html=True)

    if auto_refresh:
        time.sleep(2)
        st.rerun()

log("App started", "info")

st.markdown("""
<div style="text-align: center; padding: 20px 0 8px;">
    <h1 style="font-size: 28px; font-weight: 700; margin: 0;">
        Nexus <span style="color: #a78bfa;">AI</span>
    </h1>
    <p style="color: #666; font-size: 13px; margin-top: 4px;">Two models, one platform</p>
</div>
""", unsafe_allow_html=True)

model_choice = st.radio(
    "Model",
    ["SmAll v1", "Plus v2"],
    horizontal=True,
    label_visibility="collapsed",
)

log(f"Selected model: {model_choice}", "info")

if "generating" in st.session_state and st.session_state.generating:
    st.markdown("""
    <div class="status-bar">
        <div class="status-dot loading"></div>
        <span>Generating...</span>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="status-bar">
        <div class="status-dot"></div>
        <span>Ready</span>
    </div>
    """, unsafe_allow_html=True)

if model_choice == "SmAll v1":
    from model import Nexus
    from config import NexusConfig
    from tokenizers import Tokenizer

    @st.cache_resource(show_spinner=False)
    def load_small_model():
        log("Loading NexusConfig...", "info")
        _config = NexusConfig()
        log("Initializing Nexus model (89.8M params)...", "info")
        _model = Nexus(_config)
        log("Downloading weights from HF...", "info")
        weights_path = hf_hub_download(repo_id=REPO_SMALL, filename="weights/nexus_instruct.pt")
        log("Loading checkpoint...", "info")
        checkpoint = torch.load(weights_path, map_location=torch.device("cpu"), weights_only=False)
        _model.load_state_dict(checkpoint["model_state_dict"])
        log("Quantizing to int8...", "info")
        _model = torch.quantization.quantize_dynamic(_model, {torch.nn.Linear}, dtype=torch.qint8)
        _model.eval()
        log("Loading tokenizer...", "info")
        tokenizer_path = hf_hub_download(repo_id=REPO_SMALL, filename="data/tokenizer.json")
        _tokenizer = Tokenizer.from_file(tokenizer_path)
        log("SmAll v1 ready", "ok")
        return _model, _tokenizer, _config, torch.device("cpu")

    model, tokenizer, config, device = load_small_model()
    bos_id = tokenizer.token_to_id("<bos>") or 1
    eos_id = tokenizer.token_to_id("<eos>") or 2
else:
    config = None
    log("Plus v2: using HF Inference API", "info")

session_key = f"messages_{model_choice.replace(' ', '_')}"
if session_key not in st.session_state:
    st.session_state[session_key] = []
if "generating" not in st.session_state:
    st.session_state.generating = False

for msg in st.session_state[session_key]:
    st.chat_message(msg["role"]).write(msg["content"])

if not st.session_state[session_key]:
    if model_choice == "SmAll v1":
        welcome = "**Hello!** I'm Nexus SmAll v1 — a tiny 89.8M model built from scratch. Ask me anything (but keep expectations low 😄)"
    else:
        welcome = "**Hello!** I'm Nexus Plus v2 — a Qwen3-4B model fine-tuned with QLoRA. Smarter responses, deeper reasoning."
    with st.chat_message("assistant"):
        st.markdown(welcome)

if prompt := st.chat_input("Type a message..."):
    st.session_state[session_key].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    log(f"User: {prompt[:80]}{'...' if len(prompt)>80 else ''}", "info")
    st.session_state.generating = True
    st.rerun()

if st.session_state.generating and st.session_state[session_key]:
    last_user_msg = st.session_state[session_key][-1]["content"]

    with st.chat_message("assistant"):
        typing_placeholder = st.empty()
        typing_placeholder.markdown("""
        <div class="typing-indicator">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>
        """, unsafe_allow_html=True)

        start_time = time.time()

        if model_choice == "SmAll v1":
            log("Encoding input tokens...", "info")
            tokens = [bos_id]
            for msg in st.session_state[session_key][:-1]:
                if msg["role"] == "user":
                    tokens.extend(tokenizer.encode(f"User: {msg['content']}\nAssistant:").ids)
                elif msg["role"] == "assistant":
                    tokens.extend(tokenizer.encode(f" {msg['content']}").ids + [eos_id])
            tokens.extend(tokenizer.encode(f"User: {last_user_msg}\nAssistant:").ids)
            log(f"Input: {len(tokens)} tokens", "dim")

            input_tensor = torch.tensor([tokens[-config.max_seq_len:]], dtype=torch.long, device=device)

            log("Generating (max 128 tokens)...", "info")
            gen_start = time.time()
            with torch.no_grad():
                for i in range(128):
                    seq_len = input_tensor.shape[1]
                    if seq_len > config.max_seq_len:
                        input_tensor = input_tensor[:, -config.max_seq_len:]
                    logits = model(input_tensor, 0)
                    logits = logits[:, -1, :] / 0.2
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    input_tensor = torch.cat([input_tensor, next_token], dim=-1)
                    if next_token.item() == eos_id:
                        log(f"EOS at token {i+1}", "dim")
                        break

            elapsed = time.time() - start_time
            new_ids = input_tensor[0].tolist()[len(tokens):]
            response = tokenizer.decode(new_ids)
            for tok in ["<|assistant|>", "<|user|>", "<|system|>"]:
                response = response.replace(tok, "")
            response = response.split("<eos>")[0].split("User:")[0].replace("Assistant:", "").strip()
            if len(response) < 2:
                response = "..."
            tok_speed = len(new_ids)/elapsed if elapsed > 0 else 0
            token_info = f"⚡ {elapsed:.1f}s · {len(new_ids)} tokens · ~{tok_speed:.1f} tok/s"
            log(f"Done: {len(new_ids)} tokens in {elapsed:.1f}s ({tok_speed:.1f} tok/s)", "ok")
        else:
            client = InferenceClient(token=st.secrets.get("HF_TOKEN", None))

            messages = []
            for msg in st.session_state[session_key][:-1]:
                messages.append({"role": msg["role"], "content": msg["content"]})
            messages.append({"role": "user", "content": last_user_msg})

            log(f"Calling HF Inference API (stream)...", "info")
            try:
                response = ""
                for token in client.chat_completion(
                    model="JustScriptzz/nexus-plus-v2",
                    messages=messages,
                    max_tokens=512,
                    temperature=0.7,
                    top_p=0.8,
                    stream=True,
                ):
                    if token.choices[0].delta.content:
                        response += token.choices[0].delta.content
                elapsed = time.time() - start_time
                token_count = len(response.split())
                token_info = f"⚡ {elapsed:.1f}s · ~{token_count} tokens"
                log(f"Done: ~{token_count} tokens in {elapsed:.1f}s", "ok")
            except Exception as e:
                response = f"Error: {str(e)}"
                elapsed = time.time() - start_time
                token_info = f"❌ {elapsed:.1f}s"
                log(f"Error: {str(e)[:100]}", "err")

        typing_placeholder.empty()
        st.markdown(response)
        st.caption(token_info)

    st.session_state[session_key].append({"role": "assistant", "content": response})
    st.session_state.generating = False
    st.rerun()
