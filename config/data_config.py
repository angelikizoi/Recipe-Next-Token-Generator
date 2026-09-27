from dataclasses import dataclass, field


@dataclass
class DataConfig:
    regex_pattern: str = r""" ?<\|\p{L}++\|>|'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"""
    csv_filepath: str = 'data/full_dataset.csv'
    txt_filepath: str = 'data/recipe.txt'
    offsets_filepath:str  = 'data/recipe_offsets.npy'
    custom_tokenizer_dir: str = 'tokenizer/custom_tokenizer_files'
    hug_tokenizer: str = 'tokenizer/hug_tokenizer_files/tokenizer.json'
    special_tokens: dict[str, int] = field(default_factory=lambda: {
        " <|TITLE|>": 507,
        " <|INGREDIENTS|>": 508,
        " <|DIRECTIONS|>": 509,
        " <|EOS|>": 510,
        "<|PAD|>": 511
    })
    total_chars: int = 256  # utf-8 encoding max value
    total_merges: int = 251
    total_vocab_size: int = 512
    hug_limit_alphabet: int = 100


