import pytest

from config.data_config import DataConfig
from tokenizer.tokenizer import Tokenizer, get_pad_idx, get_vocab_size

TEXTS = [
    "chicken, rice, curry",
    "Preheat oven to 350 degrees.",
    "1/2 cup sugar, 3 eggs & 2 tbsp butter",
]


@pytest.fixture(scope="module", params=["custom", "hug"])
def tokenizer(request):
    return Tokenizer(tokenizer_type=request.param)


@pytest.mark.parametrize("text", TEXTS)
def test_roundtrip_is_lossless(tokenizer, text):
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_encode_returns_non_empty_ids(tokenizer):
    assert len(tokenizer.encode("tomato basil")) > 0


def test_ids_stay_inside_embedding_range(tokenizer):
    vocab_size = get_vocab_size(tokenizer.tokenizer_type)
    for text in TEXTS:
        assert all(0 <= i < vocab_size for i in tokenizer.encode(text))


def test_special_tokens_encode_as_single_ids():
    config = DataConfig()
    tokenizer = Tokenizer(tokenizer_type="custom", config=config)
    for token, expected_id in config.special_tokens.items():
        assert tokenizer.encode(token) == [expected_id]


def test_pad_idx_is_within_vocab():
    for tokenizer_type in ("custom", "hug"):
        assert get_pad_idx(tokenizer_type) < get_vocab_size(tokenizer_type)


def test_merges_are_applied():
    """A trained BPE must compress below one token per byte, or no merges are firing."""
    tokenizer = Tokenizer(tokenizer_type="custom")
    text = "the ingredients and the butter"
    assert len(tokenizer.encode(text)) < len(text.encode("utf-8"))
