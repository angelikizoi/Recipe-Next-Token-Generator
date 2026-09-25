from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 512
    batch_size: int = 128
    sentence_length: int = 1024 # max_seq_length=7329 with custom tokenizer
    n_layers: int = 8
    d_model: int = 256
    n_heads: int = 8
    n_epochs: int = 10
    dropout: float = 0.1
    clip_grad_norm: float = 1.0
    learning_rate: float = 1e-4
    warmup_iters: int = 1000
    min_lr_ratio: float = 0.1
    train_test_split: float = 0.9

    