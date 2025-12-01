from tokenizers.trainers import BpeTrainer
from tokenizers.models import BPE
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Split
from tokenizers import Regex
import os

from config.data_config import DataConfig

config = DataConfig()
special_tokens = list(config.special_tokens.keys())
TIKTOKEN_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"""

tokenizer = Tokenizer(BPE())
trainer = BpeTrainer(special_tokens=special_tokens, vocab_size=config.total_vocab_size)
tokenizer.pre_tokenizer = Split(Regex(TIKTOKEN_PATTERN), 'isolated')

tokenizer.train([config.txt_filepath], trainer=trainer)

try:
    os.mkdir(config.hug_tokenizer_dir)
except FileExistsError:
    pass

tokenizer.save(os.path.join(config.hug_tokenizer_dir, "tokenizer.json"))


            




