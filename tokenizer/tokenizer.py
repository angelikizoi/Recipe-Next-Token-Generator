import os
# Must be set before the `tokenizers` Rust extension spins up its internal thread
# pool; otherwise DataLoader worker processes forked afterwards (num_workers>0)
# deadlock on first use, since the pool doesn't survive fork.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from config.data_config import DataConfig
from typing import Dict, List
from collections import defaultdict
import ast
import tiktoken
import numpy as np
import regex as re
import json


class Tokenizer:
    def __init__(self, tokenizer_type='custom', config=None):
        """
        tokenizer_type: one of {"custom", "hug", "tiktoken"}
        """
        self.config = config or DataConfig()
        if tokenizer_type == 'tiktoken':
            base_enc = tiktoken.get_encoding("cl100k_base")
            self.tokenizer = tiktoken.Encoding(
                name="custom_tiktoken",
                pat_str=base_enc._pat_str,
                mergeable_ranks=base_enc._mergeable_ranks,
                special_tokens=self.config.special_tokens_tiktoken
            )
        elif tokenizer_type == 'hug':
            from tokenizers import Tokenizer
            self.tokenizer = Tokenizer.from_file(self.config.hug_tokenizer)

        elif tokenizer_type == 'custom':
            self.compiled_pattern = re.compile(self.config.regex_pattern)
            self.vocab = self._load_vocab(self.config.custom_tokenizer_dir)
            self.merges = self._load_merges(self.config.custom_tokenizer_dir)
            for pair, value in self.merges.items():
                pair = ast.literal_eval(pair)
                self.vocab[value] = self.vocab[pair[0]] + self.vocab[pair[1]]

        else:
            raise ValueError(
                f"Unknown tokenizer_type '{tokenizer_type}'. Must be one of ['custom', 'hug', 'tiktoken']."
            )
        self.tokenizer_type = tokenizer_type
        # Each backend assigns its own ids to the special tokens -- the custom scheme
        # puts them at the top of the vocabulary, HuggingFace at the bottom. Resolving
        # them per backend keeps callers (padding, EOS) from assuming one layout.
        self.special_tokens = self._resolve_special_tokens()
        self.special_ids = {i: name for name, i in self.special_tokens.items()}

    def _resolve_special_tokens(self) -> Dict[str, int]:
        if self.tokenizer_type == 'tiktoken':
            return dict(self.config.special_tokens_tiktoken)
        if self.tokenizer_type == 'custom':
            return dict(self.config.special_tokens)
        resolved = {}
        for name in self.config.special_tokens:
            token_id = self.tokenizer.token_to_id(name)
            if token_id is None:
                raise ValueError(
                    f"The HuggingFace tokenizer at {self.config.hug_tokenizer} has no "
                    f"id for {name!r}. Retrain it with `python -m tokenizer.hugging_face_tokenizer`."
                )
            resolved[name] = token_id
        return resolved

    @property
    def pad_id(self) -> int:
        return self.special_tokens["<|PAD|>"]

    @property
    def eos_id(self) -> int:
        return self.special_tokens[" <|EOS|>"]


    def encode(self, text: str) -> List[int]:
        if self.tokenizer_type == 'custom':
            text_list = re.findall(self.compiled_pattern, text)
            raw_text_ids = [list(word.encode("utf-8")) if word not in self.special_tokens else [self.special_tokens[word]] for word in text_list]
            text_ids = [token for word_ids in raw_text_ids for token in self._compress_word(word_ids)]
            return text_ids
        elif self.tokenizer_type == 'hug':
            return self.tokenizer.encode(text).ids
        else:
            return self.tokenizer.encode(text, allowed_special='all')
    

    def decode(self, text_ids: List[int], skip_special_tokens: bool = False) -> str:
        if skip_special_tokens:
            text_ids = [i for i in text_ids if i not in self.special_ids]

        if self.tokenizer_type == 'custom':
            chunks = []
            for token_id in text_ids:
                if token_id in self.special_ids:
                    chunks.append(self.special_ids[token_id].encode("utf-8"))
                else:
                    chunks.append(self.vocab.get(token_id, b""))
            return b"".join(chunks).decode(errors='replace')
        elif self.tokenizer_type == 'hug':
            return "".join(
                self.tokenizer.decode([token_id], skip_special_tokens=False)
                for token_id in text_ids
            )
        else:
            return self.tokenizer.decode(text_ids)
    
    def _compress_word(self, word: List[int]) -> List[int]:
        while len(word) > 1:
            pairs_dict = defaultdict(list)
            for i, pair in enumerate(zip(word[:-1], word[1:])):
                pairs_dict[pair].append(i)
            replacement_pair = min(pairs_dict, key=lambda p: self.merges.get(str(p), float("inf")))
            if str(replacement_pair) not in self.merges:
                break
            replacement_idx = self.merges[str(replacement_pair)]
            for i, idx in enumerate(pairs_dict[replacement_pair]):
                idx_replace = idx - i
                word[idx_replace: idx_replace+2] = [replacement_idx]
        return word


    def _load_vocab(self, directory: str) -> Dict[int, bytes]:
        with open(os.path.join(directory, 'vocabulary.bpe'), 'r') as f:
            vocab_data = f.read()
        vocab = [tuple(merge_str.split('\t')) for merge_str in vocab_data.split('\n')[:-1]]
        vocab = {int(tuple_element[0]): ast.literal_eval(tuple_element[1]) for tuple_element in vocab}
        return vocab

    def _load_merges(self, directory: str) -> Dict[tuple, int]:
        with open(os.path.join(directory, 'merges.json'), 'r') as f:
            merges = json.load(f)

        return merges


def get_vocab_size(tokenizer_type: str, config: DataConfig = None) -> int:
    """Vocab size needed by the model embedding/output layers for a given tokenizer."""
    config = config or DataConfig()
    if tokenizer_type == "tiktoken":
        return max(config.special_tokens_tiktoken.values()) + 1
    return config.total_vocab_size


def get_pad_idx(tokenizer_type: str, config: DataConfig = None) -> int:
    """Resolve <|PAD|> against the tokenizer itself.

    Assuming the custom layout here previously handed the HuggingFace runs a pad id
    of 511, which is an ordinary word in that vocabulary -- so every real occurrence
    of it was masked out of attention and dropped from the loss.
    """
    return Tokenizer(tokenizer_type=tokenizer_type, config=config).pad_id


def find_max_seq_length(tokenizer: Tokenizer, config: DataConfig) -> int:
    line_offsets = np.load(config.offsets_filepath)
    with open(config.txt_filepath, "r", encoding="utf-8") as f:
        max_length = 0
        for i in range(len(line_offsets)):
            f.seek(int(line_offsets[i]))
            text = f.readline().rstrip()
            text_ids = tokenizer.encode(text)
            max_length = max(max_length, len(text_ids))
    return max_length



if __name__ == "__main__":
    tokenizer = Tokenizer(tokenizer_type='custom')
    config = DataConfig()
    max_seq_length = find_max_seq_length(tokenizer, config)
    print(max_seq_length)
