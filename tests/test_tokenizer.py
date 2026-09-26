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


def test_special_tokens_encode_as_single_ids(tokenizer):
    for token in DataConfig().special_tokens:
        assert tokenizer.encode(token) == [tokenizer.special_tokens[token]]


def test_special_tokens_survive_a_decode_roundtrip(tokenizer):
    """Regression: custom decode silently dropped every marker, and <|TITLE|> shared
    an id with the merge ' cho', so titles decoded as 'chocolate' fragments."""
    record = (" <|TITLE|> Cake <|INGREDIENTS|> chocolate, chopped nuts"
              " <|DIRECTIONS|> Bake. <|EOS|>")
    assert tokenizer.decode(tokenizer.encode(record)) == record


def test_skip_special_tokens_removes_only_the_markers(tokenizer):
    record = " <|TITLE|> Cake <|INGREDIENTS|> sugar <|EOS|>"
    clean = tokenizer.decode(tokenizer.encode(record), skip_special_tokens=True)
    assert "<|" not in clean
    assert "Cake" in clean and "sugar" in clean


def test_special_token_ids_are_disjoint_from_ordinary_tokens(tokenizer):
    """Regression: a 252nd merge overran the vocabulary budget and collided with
    <|TITLE|>, so one id meant both a structure marker and a piece of text."""
    for token_id, name in tokenizer.special_ids.items():
        assert tokenizer.decode([token_id]) == name


def test_pad_idx_is_resolved_against_the_tokenizer(tokenizer):
    """Regression: the custom pad id (511) was used for HuggingFace too, where it is
    the ordinary word ' serving' -- masking a real word out of attention and loss."""
    assert get_pad_idx(tokenizer.tokenizer_type) == tokenizer.pad_id
    assert tokenizer.decode([tokenizer.pad_id]) == "<|PAD|>"


def test_pad_idx_is_within_vocab():
    for tokenizer_type in ("custom", "hug"):
        assert get_pad_idx(tokenizer_type) < get_vocab_size(tokenizer_type)


def test_merges_are_applied():
    """A trained BPE must compress below one token per byte, or no merges are firing."""
    tokenizer = Tokenizer(tokenizer_type="custom")
    text = "the ingredients and the butter"
    assert len(tokenizer.encode(text)) < len(text.encode("utf-8"))
