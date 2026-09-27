from tokenizers.trainers import BpeTrainer
from tokenizers.models import BPE
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Split
from tokenizers import Regex
import os

from config.data_config import DataConfig

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"""


def train_tokenizer(config=None):
    config = config or DataConfig()

    tokenizer = Tokenizer(BPE())
    # limit_alphabet is load-bearing: the full corpus contains ~628 distinct characters,
    # which on its own exceeds vocab_size and leaves zero budget for merges.
    trainer = BpeTrainer(
        special_tokens=list(config.special_tokens.keys()),
        vocab_size=config.total_vocab_size,
        limit_alphabet=config.hug_limit_alphabet,
    )
    tokenizer.pre_tokenizer = Split(Regex(GPT4_SPLIT_PATTERN), 'isolated')

    tokenizer.train([config.txt_filepath], trainer=trainer)

    os.makedirs(os.path.dirname(config.hug_tokenizer), exist_ok=True)
    tokenizer.save(config.hug_tokenizer)
    return tokenizer


if __name__ == "__main__":
    train_tokenizer()
