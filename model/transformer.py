import torch
from torch import nn
import torch.nn.functional as F



class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.n_heads = config.n_heads
        self.d_model = config.d_model
        assert self.d_model % self.n_heads == 0
        self.dk = self.d_model // self.n_heads

        self.attn = nn.Linear(self.d_model, 3 * self.d_model)
        self.fc_out = nn.Linear(self.d_model, self.d_model)
        self.fc_out.NANO_GPT_SCALE_INIT = 1
        self.dropout = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)

    def _split_heads(self, x):
        batch_size, seq_length, _ = x.size()
        return x.view(batch_size, seq_length, self.n_heads, self.dk).transpose(1, 2)

    def _combine_heads(self, x):
        batch_size, _, seq_length, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)


    def forward(self, x, attention_mask=None):
        qkv = self.attn(x) #qkv.shape=(batch_size, seq_length, 3*d_model)
        q, k, v = qkv.split(self.d_model, dim=2)
        q, k, v = self._split_heads(q), self._split_heads(k), self._split_heads(v)
        if attention_mask is not None:
            attn_mask = attention_mask[:, None, None, :]
        else:
            attn_mask = None
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = self._combine_heads(out)

        return self.resid_dropout(self.fc_out(out))
    


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.d_model, 4 * config.d_model)
        self.g = nn.GELU()
        self.fc2 = nn.Linear(4 * config.d_model, config.d_model)
        self.fc2.NANO_GPT_SCALE_INIT = 1
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x= self.g(self.fc1(x))
        x = self.fc2(x)
        return self.dropout(x)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.l1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadAttention(config)
        self.l2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.l1(x), attention_mask)
        x = x + self.mlp(self.l2(x))
        return x


class Transformer_Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.wpe = nn.Embedding(config.sentence_length, config.d_model)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.ln = nn.LayerNorm(config.d_model)
        
        self.linear = nn.Linear(config.d_model, config.vocab_size)

        self.drop = nn.Dropout(config.dropout)

        self.wte.weight = self.linear.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            # residual projections are scaled down so the residual stream variance
            # stays ~constant with depth (two per block: attention out + MLP out)
            if hasattr(module, 'NANO_GPT_SCALE_INIT'):
                std *= (2 * self.config.n_layers) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, pad_idx, idx, targets=None, attention_mask=None):
        B, T = idx.size()
        assert T <= self.config.sentence_length
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.wpe(pos)
        tok_emb = self.wte(idx)
        x = self.drop(tok_emb + pos_emb)
        for block in self.h:
            x = block(x, attention_mask)
        x = self.ln(x)
        logits = self.linear(x) #batch_size, seq_length, vocab_size
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=pad_idx)
        return logits, loss



