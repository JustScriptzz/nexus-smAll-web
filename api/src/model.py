import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from src.config import NexusConfig

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms * self.weight.float()).type_as(x)

def precompute_freqs_cis(config: NexusConfig) -> torch.Tensor:
    dim = config.dim // config.num_heads
    freqs = 1.0 / (config.rope_theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(config.max_seq_len)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)

class RotaryEmbedding(nn.Module):
    def __init__(self, config: NexusConfig):
        super().__init__()
        self.freqs_cis = precompute_freqs_cis(config)

    def forward(self, x: torch.Tensor, start_pos: int = 0):
        _, seq_len, _, head_dim = x.shape
        freqs_cis = self.freqs_cis[start_pos:start_pos+seq_len, :head_dim//2].to(x.device)
        freqs_cis = freqs_cis.view(1, seq_len, 1, head_dim//2)
        
        x_shaped = x.float().reshape(*x.shape[:-1], -1, 2)
        x_complex = torch.complex(x_shaped[..., 0], x_shaped[..., 1])
        
        x_rotated = x_complex * freqs_cis
        
        x_out = torch.stack([x_rotated.real, x_rotated.imag], dim=-1).reshape_as(x_shaped)
        
        return x_out.reshape_as(x).type_as(x)

class Attention(nn.Module):
    def __init__(self, config: NexusConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        if self.num_kv_heads is None:
            self.num_kv_heads = config.num_heads
        self.head_dim = config.dim // config.num_heads
        self.num_kv_groups = config.num_heads // self.num_kv_heads

        self.wq = nn.Linear(config.dim, config.dim, bias=False)
        self.wk = nn.Linear(config.dim, self.head_dim * self.num_kv_heads, bias=False)
        self.wv = nn.Linear(config.dim, self.head_dim * self.num_kv_heads, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.rotary = RotaryEmbedding(config)

    def forward(self, x: torch.Tensor, start_pos: int = 0, mask: Optional[torch.Tensor] = None):
        bsz, seqlen, _ = x.shape

        q = self.wq(x).view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.rotary(q, start_pos)
        k = self.rotary(k, start_pos)

        if self.num_kv_groups > 1:
            k = k[:, :, None, :, :].expand(bsz, self.num_kv_heads, self.num_kv_groups, seqlen, self.head_dim)
            k = k.reshape(bsz, self.num_heads, seqlen, self.head_dim)
            v = v[:, :, None, :, :].expand(bsz, self.num_kv_heads, self.num_kv_groups, seqlen, self.head_dim)
            v = v.reshape(bsz, self.num_heads, seqlen, self.head_dim)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        if mask is not None:
            attn_weights = attn_weights + mask

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(q)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(attn_output)

class FeedForward(nn.Module):
    def __init__(self, config: NexusConfig):
        super().__init__()
        hidden_dim = int(2 * config.ff_dim / 3)
        hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)

        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
    def __init__(self, config: NexusConfig):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)
        self.ff_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int = 0, mask: Optional[torch.Tensor] = None):
        h = x + self.attention(self.attention_norm(x), start_pos, mask)
        out = h + self.feed_forward(self.ff_norm(h))
        return out

class Nexus(nn.Module):
    def __init__(self, config: NexusConfig):
        super().__init__()
        self.config = config

        self.token_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.dim, config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.token_embeddings.weight = self.output.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, start_pos: int = 0):
        _, seqlen = input_ids.shape

        mask = torch.full((1, 1, seqlen, start_pos + seqlen), float('-inf'),
                          dtype=torch.float32, device=input_ids.device)
        mask = torch.triu(mask, diagonal=start_pos + 1).type_as(input_ids)

        x = self.token_embeddings(input_ids)

        for layer in self.layers:
            x = layer(x, start_pos, mask)

        x = self.norm(x)
        logits = self.output(x)

        return logits

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int,
                 temperature: float = 0.7, top_k: int = 50, top_p: float = 0.9):
        self.eval()
        generated = []

        for _ in range(max_new_tokens):
            seq_len = input_ids.shape[1]
            if seq_len > self.config.max_seq_len:
                input_ids = input_ids[:, -self.config.max_seq_len:]

            with torch.no_grad():
                logits = self(input_ids, 0)
                logits = logits[:, -1, :] / temperature

                if top_k > 0:
                    top_k_values, _ = torch.topk(logits, top_k)
                    min_top_k = top_k_values[:, -1].unsqueeze(-1)
                    logits = torch.where(logits < min_top_k,
                                         torch.full_like(logits, float('-inf')), logits)

                if top_p > 0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 0] = False
                    indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
                    indices_to_remove = indices_to_remove.scatter(1, sorted_indices,
                                                                  sorted_indices_to_remove)
                    logits = torch.where(indices_to_remove,
                                         torch.full_like(logits, float('-inf')), logits)

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=-1)
            generated.append(next_token.item())

        return generated, input_ids

def create_nexus_model():
    from config import nexus_config
    config = nexus_config

    model = Nexus(config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"[Nexus SmAll] Model created with {total_params/1e6:.1f}M parameters "
          f"({trainable_params/1e6:.1f}M trainable)")

    return model, config