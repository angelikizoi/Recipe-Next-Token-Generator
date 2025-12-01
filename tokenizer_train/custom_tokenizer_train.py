from config.data_config import DataConfig
from typing import Dict, Generator, List, Tuple
from multiprocessing import Pool
from collections import Counter
from functools import partial
import pandas as pd
import regex as re
import os
import pickle
import tempfile
import json
import time
import cmsketch
import gc

TIKTOKEN_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"""
compiled_pattern = re.compile(TIKTOKEN_PATTERN)


def chunks(file_name: str, size: int=100000) -> Generator[pd.DataFrame, None, None]:
    df_chunks = pd.read_csv(file_name, usecols=["title", "NER", "directions"], chunksize=size)
    for chunk in df_chunks:
        yield chunk
        del chunk
        gc.collect()

def transform_df_chunk(df_chunk: pd.DataFrame) -> List[List[int]]: 
    df_chunk['NER'] = df_chunk['NER'].str.strip('[]').str.replace('"', '')
    df_chunk['directions'] = df_chunk['directions'].str.strip('[]').str.replace('"', '').str.replace('.,', '.')
    chunk_recipe_list = []
    for row in df_chunk.itertuples():
        text = f"{row[1]} {row[3]} {row[2]} " # {title} {ingredients} {directions} 
        # Step 1 : Creating a list of tokens ('words') by applying gpt-4 regex pattern
        tokens = re.findall(compiled_pattern, text)
        # Step 2: Following Tiktoken tokenizer we convert each character to byte. So each word of
        # the text is represented as a sequence (list) of its letter bytes.
        chunk_recipe_list.extend([list(t.encode('utf-8')) for t in tokens])
    del df_chunk
    gc.collect()
    return chunk_recipe_list


def stats_pair_cms(chunk: List[List[int]]) -> Dict[str, int]:
    # Step 3: Using CountMinSketch algorithm compute all pairs' frequencies and return the top 100
    # from each chunk.
    sketch = cmsketch.CountMinSketchStr(1000000, 5)
    candidate_pairs = set()
    for word in chunk:
        for pair in zip(word[:-1], word[1:]):
            sketch.insert(str(pair))
            candidate_pairs.add(str(pair))
    top_freq_pair_dict = {key: value for (key, value) in sketch.top_k(100, candidate_pairs)}
    sketch.clear()
    del candidate_pairs
    gc.collect()
    return top_freq_pair_dict


def merge(chunk: List[List[int]]) -> List[List[int]]:
    # Step 4: In each chunk replace all consecutive occurrences
    # of the most frequent pair with the new integer token idx
    max_pair = max(merges, key=merges.get)
    max_id = merges[max_pair]

    for word in chunk:
        i = 0
        for pair in zip(word[:-1], word[1:]):
            if pair == max_pair:
                word[i:i+2] = [max_id]
            else:
                i += 1
    return chunk


def process_chunk(chunks_iter: Generator[pd.DataFrame, None, None] | Generator[List[List[int]], None, None], i:int) -> Tuple[Dict[str, int], str]:
    if i == 0:
        transformed = transform_df_chunk(chunks_iter)
    else: 
        transformed = merge(chunks_iter)
    del chunks_iter
    gc.collect()
    top_freq_pair_dict = stats_pair_cms(transformed)
    temp_path = tempfile.NamedTemporaryFile(prefix=f"chunk_{i}_", suffix=".pkl", delete=False)
    with open(temp_path.name, "wb") as f:
        pickle.dump(transformed, f, protocol=pickle.HIGHEST_PROTOCOL)
    del transformed
    gc.collect()
    return top_freq_pair_dict, temp_path.name


def load_next_chunks(temp_files: List[str]) -> Generator[List[List[int]], None, None]:
    for f in temp_files:
        with open(f, 'rb') as infile:
            yield pickle.load(infile)
        os.remove(f)
        gc.collect()


if __name__ == '__main__':
    config = DataConfig()

    merges = {}
    current_idx = config.total_chars
    temp_files = []

    for i in range(config.total_merges):
        start_time = time.time()
        
        if i == 0:
            chunks_iter = chunks(config.csv_filepath)
        else:
            chunks_iter = load_next_chunks(temp_files)
        with Pool() as pool:
            total_counter = Counter()
            process_chunk_partial = partial(process_chunk, i=i)
            results = pool.imap(process_chunk_partial, chunks_iter)
            temp_files = []
            for top_freq_pair_dict, temp_path in results:
                total_counter.update(top_freq_pair_dict)
                temp_files.append(temp_path)
            temp_files.sort()
            most_frequent_pair = eval(total_counter.most_common(1)[0][0])
            merges.update({most_frequent_pair: current_idx})
            del total_counter, results
            gc.collect()
            current_idx += 1
            print(f"Merge {i} ended in {(time.time() - start_time)/60:.2f} minutes.")

    try:
        os.mkdir(config.custom_tokenizer_dir)
    except FileExistsError:
        pass

    sorted_merges = {str(k): v for k, v in sorted(merges.items(), key=lambda item: item[1])}
    with open(os.path.join(config.custom_tokenizer_dir, "merges.json"), "w") as f:
        json.dump(sorted_merges, f)
    
    vocab = {idx: bytes([idx]) for idx in range(config.total_chars)}
    with open(os.path.join(config.custom_tokenizer_dir, "vocabulary.bpe"), "w") as f:
        for key, value in vocab.items():
            f.write(f"{key}\t{value}\n")



  