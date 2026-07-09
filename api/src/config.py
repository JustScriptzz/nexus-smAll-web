from dataclasses import dataclass

@dataclass
class NexusConfig:
    vocab_size: int = 50304
    max_seq_len: int = 512
    dim: int = 768
    num_layers: int = 10
    num_heads: int = 12
    num_kv_heads: int = 4
    multiple_of: int = 256
    ff_dim: int = 2048
    norm_eps: float = 1e-6
    dropout: float = 0.0
    
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    warmup_steps: int = 500
    max_steps: int = 100000
    
    rope_theta: float = 500000.0
    
    use_flash_attention: bool = True
    
    data_path: str = "data"
    save_dir: str = "weights"
    
    seed: int = 42

nexus_config = NexusConfig()