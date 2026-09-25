from config.data_config import DataConfig
import pandas as pd
import numpy as np
import os


class DataPreprocessor:
    def __init__(self, config=None):
        config = config or DataConfig()
        self.csv_filepath = config.csv_filepath
        self.txt_filepath = config.txt_filepath
        self.special_tokens = list(config.special_tokens.keys())
        self.line_offsets = []
        self.chunk_size_input = 20000

        if not os.path.exists(self.txt_filepath):
            pipeline = self._load(self._transform(self._extract()))
            for _ in pipeline:
                pass
            np.save(config.offsets_filepath, np.array(self.line_offsets, dtype=np.int64))

    def _extract(self):
        print('Extracting...')
        chunks_iter = pd.read_csv(self.csv_filepath, usecols=["title", "NER", "directions"], chunksize=self.chunk_size_input)
        for chunk in chunks_iter:
            yield chunk

    def _transform(self, chunks):
        print('Transforming...')
        for df in chunks:
            df['NER'] = df['NER'].str.strip('[]').str.replace('"', '')
            df['directions'] = df['directions'].str.strip('[]').str.replace('"', '').str.replace('.,', '.')
            yield df

    def _load(self, chunks):
        print('Loading...')
        offset = 0
        self.line_offsets.append(offset)
        for df in chunks:
            with open(self.txt_filepath, "a") as f:
                for row in df.itertuples():
                    text = f"{self.special_tokens[0]} {row[1]}{self.special_tokens[1]} {row[3]}{self.special_tokens[2]} {row[2]}{self.special_tokens[3]}\n"
                    offset += len(text.encode('utf-8'))
                    self.line_offsets.append(offset)
                    f.write(text)
            yield True


if __name__ == '__main__':
    datapreprocessor = DataPreprocessor()


