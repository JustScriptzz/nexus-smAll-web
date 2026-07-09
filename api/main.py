import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from tokenizers import Tokenizer
from src.model import Nexus
from src.config import NexusConfig
from huggingface_hub import hf_hub_download

app = FastAPI(title="Nexus SmAll v1 API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

REPO = "JustScriptzz/nexus-smAll-v1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading Nexus SmAll v1 on {device}...")

config = NexusConfig()
model = Nexus(config)

weights_path = hf_hub_download(repo_id=REPO, filename="weights/nexus_instruct.pt")
checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()
print("Model loaded")

tokenizer_path = hf_hub_download(repo_id=REPO, filename="data/tokenizer.json")
tokenizer = Tokenizer.from_file(tokenizer_path)
bos_id = tokenizer.token_to_id("<bos>") or 1
eos_id = tokenizer.token_to_id("<eos>") or 2
print("Tokenizer loaded")


class ChatRequest(BaseModel):
    message: str
    history: list = []

class ChatResponse(BaseModel):
    response: str


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    tokens = [bos_id]
    for turn in req.history:
        if turn.get("role") == "user":
            tokens.extend(tokenizer.encode(f"User: {turn['content']}\nAssistant:").ids)
        elif turn.get("role") == "assistant":
            tokens.extend(tokenizer.encode(f" {turn['content']}").ids + [eos_id])

    tokens.extend(tokenizer.encode(f"User: {req.message}\nAssistant:").ids)

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

    return ChatResponse(response=response)


@app.get("/health")
def health():
    return {"status": "ok"}
