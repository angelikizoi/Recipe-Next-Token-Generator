import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

from config.data_config import DataConfig
from tokenizer.tokenizer import Tokenizer



class TokenizedDataset(Dataset):
    def __init__(self, line_offsets, max_seq_length, tokenizer_type="custom", config=None):
        self.config = config or DataConfig()
        self.line_offsets = line_offsets[:-1]
        self.tokenizer = Tokenizer(tokenizer_type)
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.line_offsets)

    def __getitem__(self, index):
        with open(self.config.txt_filepath, "r", encoding="utf-8") as f:
            f.seek(int(self.line_offsets[index]))
            text = f.readline().rstrip()
        raw_text_ids = self.tokenizer.encode(text)
        if len(raw_text_ids) > self.max_seq_length:
            text_ids = raw_text_ids[:self.max_seq_length - 1] 
            text_ids.append(raw_text_ids[-1])
        else:
            text_ids = raw_text_ids
        src = torch.tensor(text_ids[:-1], dtype=torch.long)
        tgt = torch.tensor(text_ids[1:], dtype=torch.long)
        return src, tgt



class SentenceLenSampler(Sampler):
    def __init__(self, indices, lengths, batch_size):
        self.batch_size = batch_size
        self.lengths = lengths
        self.indices = indices


    def __iter__(self):
        step = 100 * self.batch_size
        n = len(self.indices)
        shuffled_indices = np.random.permutation(self.indices)
        for i in range(0, n, step):
            chunk = shuffled_indices[i : i + step]
            step_lengths = self.lengths[chunk]
            sorted_length_idx = np.argsort(step_lengths)
            pool = chunk[sorted_length_idx]
            for j in range(0, len(pool), self.batch_size):
                batch = pool[j : j + self.batch_size]
                if len(batch) == self.batch_size:  # drop_last=True
                    yield batch.tolist()

    def __len__(self):
        return len(self.indices) // self.batch_size



def collate_pad_fn(batch, pad_token_id):
    input_seqs = [item[0] for item in batch]
    label_seqs = [item[1] for item in batch]

    input_padded = pad_sequence(
        input_seqs, batch_first=True, padding_value=pad_token_id
    )
    label_padded = pad_sequence(
        label_seqs, batch_first=True, padding_value=pad_token_id
    )

    return {"input_ids": input_padded, "labels": label_padded}

