import math

import pytest
import torch

from config.model_config import ModelConfig
from model.transformer import Transformer_Decoder


@pytest.fixture
def config():
    return ModelConfig(n_layers=2, d_model=64, n_heads=4, vocab_size=512)


@pytest.fixture
def model(config):
    torch.manual_seed(0)
    return Transformer_Decoder(config).eval()


def test_forward_output_shape(model, config):
    idx = torch.randint(0, config.vocab_size, (2, 16))
    logits, loss = model(config.vocab_size - 1, idx)
    assert logits.shape == (2, 16, config.vocab_size)
    assert loss is None


def test_loss_is_returned_with_targets(model, config):
    idx = torch.randint(0, config.vocab_size, (2, 16))
    _, loss = model(config.vocab_size - 1, idx, targets=idx)
    assert loss.ndim == 0 and loss.item() > 0


def test_attention_is_causal(model, config):
    """Changing the final token must not alter logits at earlier positions."""
    idx = torch.randint(0, config.vocab_size, (1, 12))
    altered = idx.clone()
    altered[0, -1] = (altered[0, -1] + 1) % config.vocab_size

    logits_a, _ = model(config.vocab_size - 1, idx)
    logits_b, _ = model(config.vocab_size - 1, altered)

    assert torch.allclose(logits_a[:, :-1], logits_b[:, :-1], atol=1e-5)


def test_input_and_output_embeddings_are_tied(model):
    assert model.wte.weight is model.linear.weight


def test_initial_loss_matches_uniform_distribution(config):
    """A correctly initialised LM starts at ~ln(vocab_size); a bad init blows this up."""
    torch.manual_seed(0)
    model = Transformer_Decoder(config).eval()
    # targets must be shifted by one, as in training: unshifted targets leak the
    # answer through the tied embeddings and depress the loss below ln(vocab_size).
    seq = torch.randint(0, config.vocab_size - 1, (8, 33))
    _, loss = model(config.vocab_size - 1, seq[:, :-1], targets=seq[:, 1:])
    assert loss.item() == pytest.approx(math.log(config.vocab_size), abs=0.25)


def test_token_and_positional_embeddings_share_a_scale(config):
    """Regression test: wpe once defaulted to std=1.0, ~28x the token embeddings."""
    torch.manual_seed(0)
    model = Transformer_Decoder(config)
    ratio = (model.wpe.weight.std() / model.wte.weight.std()).item()
    assert 0.5 < ratio < 2.0


def test_residual_stream_is_preserved_across_blocks(config):
    """Pre-LN must normalise the branch input, not the residual stream itself."""
    torch.manual_seed(0)
    model = Transformer_Decoder(config).eval()
    x = model.wte(torch.randint(0, config.vocab_size, (2, 16))) + model.wpe(torch.arange(16))

    norms = []
    for block in model.h:
        x = block(x)
        norms.append(x.norm(dim=-1).mean().item())

    assert norms[-1] > norms[0]


def test_dropout_only_active_in_training_mode(config):
    torch.manual_seed(0)
    model = Transformer_Decoder(ModelConfig(**{**config.__dict__, "dropout": 0.5}))
    idx = torch.randint(0, config.vocab_size, (2, 16))

    model.train()
    assert not torch.allclose(model(511, idx)[0], model(511, idx)[0])

    model.eval()
    assert torch.allclose(model(511, idx)[0], model(511, idx)[0])
