# Recipe GPT

[![tests](https://github.com/angelikizoi/Recipe-Next-Token-Generator/actions/workflows/tests.yml/badge.svg)](https://github.com/angelikizoi/Recipe-Next-Token-Generator/actions/workflows/tests.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A GPT-2-style decoder-only Transformer, trained from scratch to generate cooking recipes from a short list of ingredients. Includes a custom byte-pair-encoding tokenizer built from first principles, a HuggingFace-compatible tokenizer, and full MLflow experiment tracking.

Trained on the [Recipe 2M+ dataset](https://www.kaggle.com/datasets/wilmerarltstrmberg/recipe-dataset-over-2m) (2M+ recipes).

## Pipeline

The project is organized as a linear ML pipeline, one package per stage:

```
config/       Typed dataclass configs (model hyperparameters, data paths, special tokens)
data/         Raw CSV -> flat text corpus, offset index, PyTorch Dataset/Sampler/collate
tokenizer/    Two interchangeable tokenizers: a from-scratch BPE trainer, and a HuggingFace BPE trainer
model/        The Transformer_Decoder itself (causal self-attention, tied embeddings, GPT-2-style init)
training/     Training/eval loop, MLflow logging, loss & perplexity curves
inference/    Top-k sampling generation, loadable from any MLflow run
results/      Loss/perplexity curves, sample generations, and metrics from the latest training run
```

Each stage is a runnable module:

```bash
uv sync                                   # install dependencies

python -m data.preprocess                 # CSV -> data/recipe.txt + offset index
python -m tokenizer.hugging_face_tokenizer # trains the HuggingFace BPE tokenizer
python -m tokenizer.custom_tokenizer_train # (optional) trains the from-scratch BPE tokenizer

mlflow server --host 127.0.0.1 --port 5000 &  # tracking server the scripts below expect
python -m training.train                  # trains the model, writes results/

python -m inference.generate --prompt "chicken, rice, curry"

pytest tests/                             # unit tests (no dataset required)
```

Model and data hyperparameters live in [config/model_config.py](config/model_config.py) and [config/data_config.py](config/data_config.py) — nothing is hardcoded in the pipeline scripts themselves.

## Model

* Decoder-only Transformer, GPT-2-style (causal self-attention, pre-LN blocks, tied input/output embeddings, GPT-2 init with depth-scaled residual projections)
* Configurable depth/width via `ModelConfig` — the tracked run below used 8 layers, `d_model=256`, 8 heads (~6.8M params)
* Trained with AdamW, gradient clipping, dropout, bf16 autocast, and a nanoGPT-style linear-warmup / cosine-decay learning-rate schedule
* Runs are seeded end-to-end (NumPy + Torch) and the best epoch by validation loss is restored before reporting results
* A length-bucketed batch sampler (`SentenceLenSampler`) groups similarly-sized recipes together per batch to minimize padding waste

## Tokenizer

Two BPE tokenizers are supported, sharing the same special-token scheme (`<|TITLE|>`, `<|INGREDIENTS|>`, `<|DIRECTIONS|>`, `<|EOS|>`, `<|PAD|>`):

* **Custom** ([tokenizer/custom_tokenizer_train.py](tokenizer/custom_tokenizer_train.py)) — BPE merge-learning implemented from scratch, parallelized across CPU cores with a Count-Min Sketch for approximate pair-frequency counting over the full 2M+ recipe corpus without materializing exact counts in memory.
* **HuggingFace** ([tokenizer/hugging_face_tokenizer.py](tokenizer/hugging_face_tokenizer.py)) — the same vocabulary scheme trained via `tokenizers.BpeTrainer`, used as the default at training/inference time for speed.

## Experiment tracking

Every training run logs to MLflow: hyperparameters, per-epoch train/val loss & perplexity, gradient norms, model checkpoints per epoch, and the generated result artifacts (loss/perplexity curves, sample generations). See `training/train.py`.

## Results

<!-- RESULTS_PLACEHOLDER -->

## Engineering notes

Building this from scratch surfaced a number of bugs that are easy to miss because
nothing crashes — the model simply trains worse. They are documented here because
finding them was most of the work:

* **Positional embeddings initialised 28× too large.** `_init_weights` applied the
  GPT-2 residual-scaling factor but never the underlying `normal_(std=0.02)` init, so
  every module silently kept PyTorch's default. `nn.Embedding` defaults to N(0, 1),
  which left `wpe` at std 1.0 against a tied `wte` at 0.036 — at initialisation the
  positional signal drowned out token identity ~28:1. The tell: initial loss should
  equal `ln(vocab_size)` for a correctly initialised LM. It now does (6.29 vs 6.24).
* **The pre-LN block normalised the residual stream itself** (`x = self.l1(x)`) rather
  than only the branch input (`x = x + attn(self.l1(x))`). This re-normalises away
  each block's contribution, severing the identity path that makes deep transformers
  trainable. Measured across 8 blocks, the residual norm was pinned flat at ~16.4
  instead of growing.
* **The HuggingFace tokenizer deadlocked the DataLoader.** Its Rust thread pool does
  not survive `fork`, so worker processes intermittently hung forever on the first
  batch (~50% of runs). Fixed with `multiprocessing_context="spawn"`, which also
  required replacing a `lambda` collate function with `functools.partial` to keep it
  picklable.
* **Train/val splits were not reproducible.** Only `torch.manual_seed` was set;
  `np.random.permutation` drove both the split and the batch shuffling, so every run
  silently trained on different data — making runs incomparable.
* **Only the final epoch's weights were ever used.** Training logged a checkpoint per
  epoch but always reported the last one, even when validation loss had clearly
  bottomed out earlier. Best-checkpoint selection is now tracked and restored.
* **`targets.view(-1)` crashed on any non-contiguous target tensor** — latent, because
  the dataset happened to build targets as fresh tensors. Caught by a unit test.

## Roadmap

* Learning-rate and regularisation sweep now that the architecture bugs are fixed
* Further scaling of the Transformer and tokenizer vocabulary
* A small web UI (Streamlit) for interactive recipe generation

---

*Built as a from-scratch NLP project: tokenizer, model, training loop, and experiment tracking all implemented without relying on a pre-built GPT implementation.*
