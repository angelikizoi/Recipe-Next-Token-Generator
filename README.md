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

8 layers, `d_model=256`, 8 heads (~6.8M params), vocab 512, 10 epochs over 2M+ recipes
on a single RTX 5060 Ti. Both tokenizers train under an identical configuration and seed.

| tokenizer | val loss | val perplexity | tokens/byte | **bits/byte** |
|---|---|---|---|---|
| custom BPE | 1.016 | 2.76 | 0.426 | **0.624** |
| HuggingFace BPE | 1.132 | 3.10 | 0.385 | **0.629** |

**Read the last column, not the perplexity.** Per-token perplexity is not comparable
across tokenizers: the custom BPE learned 251 merges against HuggingFace's 413, so it
emits finer-grained tokens that are each individually easier to predict. That alone
accounts for nearly all of its apparent 12% perplexity advantage. Normalised to
bits-per-byte — which measures compression of the same underlying text — the two are
within 0.8% of each other. **A BPE implemented from first principles matches a
production library's implementation to within a percent**, which is the result this
project was built to test.

Validation loss decreased monotonically for both runs, every epoch, with no instability
spikes; both were still improving at epoch 10, so these numbers are a floor rather than
a converged ceiling.

| | custom | HuggingFace |
|---|---|---|
| loss | ![custom loss](results/custom/loss_curve.png) | ![hug loss](results/hug/loss_curve.png) |
| perplexity | ![custom ppl](results/custom/perplexity_curve.png) | ![hug ppl](results/hug/perplexity_curve.png) |

Sample generations (full set in [results/](results/)), prompted with a bare ingredient list:

> **chicken, rice, curry** rice, Kale, cream of potato soup, quick-covered rotisserie
> chicken, rice — Bring pot of water to boil. Add rice to boiling water and cook 25
> minutes. In a bowl, mix meat with soup, kale, and chicken. […]

> **chicken, rice, curry** powder artichoke bites, green olive oil, garlic, salt, curry
> powder, scallions, marjoram, shallots, lemon — in a large pot, heat oil and saute
> garlic until translucent. add salt, sugar, curry powder and roasted scallions […]

At 6.8M parameters the model produces fluent recipe-shaped text with plausible
ingredient-to-method consistency, but it does not track quantities or step ordering
reliably across a whole recipe.

### Effect of the fixes

The same configuration before the initialisation and residual-path bugs described below
were fixed:

| | custom | HuggingFace |
|---|---|---|
| before | val 2.29 / ppl 9.84, diverged at epoch 5 | val 2.00 / ppl 7.39, oscillating |
| after | **val 1.02 / ppl 2.76**, monotonic | **val 1.13 / ppl 3.10**, monotonic |

Perplexity improved 3.6× and 2.4× respectively, and the training instability that
prompted the investigation disappeared entirely — it was an architecture bug, not a
learning-rate problem.

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

* Train past 10 epochs — validation loss was still falling for both runs when training stopped
* Learning-rate and regularisation sweep, now that the architecture bugs are no longer the bottleneck
* Further scaling of the Transformer and of the tokenizer vocabulary (512 is small; the
  bits-per-byte comparison should be repeated at a larger vocab)
* A small web UI (Streamlit) for interactive recipe generation

---

*Built as a from-scratch NLP project: tokenizer, model, training loop, and experiment tracking all implemented without relying on a pre-built GPT implementation.*
