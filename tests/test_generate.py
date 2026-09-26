import torch
import torch.nn as nn

from config.model_config import ModelConfig
from inference.generate import generate
from tokenizer.tokenizer import Tokenizer


class ScriptedModel(nn.Module):
    """Emits `script` one token per step, then repeats the last one forever."""

    def __init__(self, vocab_size, script, context_window=64):
        super().__init__()
        self.config = ModelConfig(vocab_size=vocab_size, sentence_length=context_window)
        self.script = script
        self.calls = 0

    def forward(self, pad_idx, idx, targets=None, attention_mask=None):
        batch, seq_len = idx.shape
        logits = torch.zeros(batch, seq_len, self.config.vocab_size)
        nxt = self.script[min(self.calls, len(self.script) - 1)]
        logits[:, -1, nxt] = 100.0
        self.calls += 1
        return logits, None


def _tokenizer():
    return Tokenizer(tokenizer_type="custom")


def test_generation_stops_at_eos():
    tok = _tokenizer()
    filler = tok.encode(" salt")[0]
    model = ScriptedModel(tok.pad_id + 1, [filler, filler, tok.eos_id])

    out = generate(model, tok, tok.pad_id, torch.device("cpu"), "cake",
                   num_return_sequences=1, top_k=1, skip_special_tokens=False)

    assert out[0].endswith("<|EOS|>")
    # three sampling steps: two fillers then EOS, and no further calls
    assert model.calls == 3


def test_generation_is_capped_by_the_context_window():
    tok = _tokenizer()
    filler = tok.encode(" salt")[0]
    model = ScriptedModel(tok.pad_id + 1, [filler], context_window=24)

    out = generate(model, tok, tok.pad_id, torch.device("cpu"), "cake",
                   num_return_sequences=1, top_k=1)

    assert len(tok.encode(out[0])) <= 24


def test_max_length_cannot_exceed_the_context_window():
    tok = _tokenizer()
    filler = tok.encode(" salt")[0]
    model = ScriptedModel(tok.pad_id + 1, [filler], context_window=16)

    out = generate(model, tok, tok.pad_id, torch.device("cpu"), "cake",
                   num_return_sequences=1, max_length=9999, top_k=1)

    assert len(tok.encode(out[0])) <= 16


def test_skip_special_tokens_controls_marker_visibility():
    tok = _tokenizer()
    model = ScriptedModel(tok.pad_id + 1, [tok.eos_id])
    args = (model, tok, tok.pad_id, torch.device("cpu"), "cake")

    raw = generate(*args, num_return_sequences=1, top_k=1, skip_special_tokens=False)[0]
    clean = generate(*args, num_return_sequences=1, top_k=1, skip_special_tokens=True)[0]

    assert "<|EOS|>" in raw
    assert "<|" not in clean
