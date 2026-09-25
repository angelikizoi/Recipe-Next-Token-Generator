import argparse
import functools
import json
import math
import os
import time
import urllib.request
import numpy as np
import mlflow
import torch
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm


from model.transformer import Transformer_Decoder
from data.dataset import TokenizedDataset, SentenceLenSampler, collate_pad_fn
from config.model_config import ModelConfig
from config.data_config import DataConfig
from tokenizer.tokenizer import Tokenizer, get_vocab_size, get_pad_idx
from inference.generate import generate
from training.visualize import loss_plot, ppl_plot

TOKENIZER_TYPES = ("custom", "hug", "tiktoken")
RESULTS_ROOT = "results"


def load_data(model_config, data_config, mode, pad_idx, tokenizer_type):
    line_offsets = np.load(data_config.offsets_filepath)
    lengths = (line_offsets[1:] - line_offsets[:-1]).astype(np.int64)
    n = len(lengths)
    indices = np.random.permutation(n)
    if mode == 'train':
        sampler_indices = indices[:int(model_config.train_test_split*n)]
    else:
        sampler_indices = indices[int(model_config.train_test_split*n):]

    data = TokenizedDataset(line_offsets, model_config.sentence_length, tokenizer_type=tokenizer_type)
    return DataLoader(
        data,
        batch_sampler=SentenceLenSampler(sampler_indices, lengths, model_config.batch_size),
        collate_fn=functools.partial(collate_pad_fn, pad_token_id=pad_idx),
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
        # 'spawn' avoids a fork-time deadlock: the HF `tokenizers` Rust extension's
        # internal thread pool doesn't reliably survive being forked mid-init, which
        # intermittently hung worker processes under the default 'fork' context.
        multiprocessing_context="spawn",
    )

def get_lr(it, warmup_iters, lr_decay_iters, min_lr, max_lr):
    # nanoGPT-style schedule: linear warmup, then cosine decay to min_lr, then constant min_lr.
    if it < warmup_iters:
        return max_lr * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def train(model, train_loader, optimizer, model_config, epoch, device, pad_idx, iters_per_epoch, total_iters):
    model.train()
    t0 = time.time()
    train_loss = 0
    train_cnt = 0
    train_iterator = tqdm(train_loader, desc=f"Training Epoch {epoch:02d}")
    train_total_batches = len(train_loader)
    min_lr = model_config.learning_rate * model_config.min_lr_ratio
    for batch_step, batch in enumerate(train_iterator):
        it = (epoch - 1) * iters_per_epoch + batch_step
        lr = get_lr(it, model_config.warmup_iters, total_iters, min_lr, model_config.learning_rate)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        src = batch["input_ids"].to(device, non_blocking=True)
        tgt = batch["labels"].to(device, non_blocking=True)
        src_mask = src != pad_idx
        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits, loss = model(pad_idx, src, tgt, src_mask)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=model_config.clip_grad_norm)
        optimizer.step()
        train_loss += loss.item()
        train_cnt += 1
        if batch_step % 200 == 0:
            step = (batch_step // 200) + (epoch - 1) * (train_total_batches//200 + 1)
            mlflow.log_metric("loss", loss.item(), step=step)
            mlflow.log_metric("lr", lr, step=step)
    torch.cuda.synchronize()
    t1 = time.time()
    train_dt = (t1 - t0) / 60
    avg_train_loss = train_loss / train_cnt
    mlflow.log_metric("train_dt", train_dt, step=epoch)
    mlflow.log_metric("avg_train_loss", avg_train_loss, step=epoch)
    mlflow.log_metric("train_ppl", math.exp(avg_train_loss), step=epoch)
    mlflow.log_metric("norm", norm.item(), step=epoch)
    print(f"epoch: {epoch}, train_loss: {avg_train_loss:.4f}, training_time: {train_dt}mins")
    return avg_train_loss, math.exp(avg_train_loss)

def evaluate(model, val_loader, epoch, device, pad_idx):
    model.eval()
    t0 = time.time()
    val_loss = 0
    val_cnt = 0
    val_iterator = tqdm(val_loader, desc=f"Validating Epoch {epoch:02d}")

    with torch.no_grad():
        for batch in val_iterator:
            src = batch["input_ids"].to(device, non_blocking=True)
            tgt = batch["labels"].to(device, non_blocking=True)
            src_mask = src != pad_idx
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits, loss = model(pad_idx, src, tgt, src_mask)
            val_loss += loss.item()
            val_cnt += 1
    torch.cuda.synchronize()
    t1 = time.time()
    avg_val_loss = val_loss / val_cnt
    val_dt = (t1 - t0) / 60
    mlflow.log_metric("val_dt", val_dt, step=epoch)
    mlflow.log_metric("avg_val_loss", avg_val_loss, step=epoch)
    mlflow.log_metric("val_ppl", math.exp(avg_val_loss), step=epoch)
    print(f"epoch: {epoch}, val_loss: {avg_val_loss:.4f}, val_time: {val_dt}mins")
    return avg_val_loss, math.exp(avg_val_loss)


def check_tracking_server(tracking_uri):
    """Fail fast: without this, an unreachable server just stalls in silent retries."""
    try:
        urllib.request.urlopen(f"{tracking_uri}/health", timeout=5)
    except Exception as exc:
        raise SystemExit(
            f"Cannot reach the MLflow tracking server at {tracking_uri} ({exc}).\n"
            f"Start it first:  mlflow server --host 127.0.0.1 --port 5000"
        )


def run_training(tokenizer_type, tracking_uri="http://localhost:5000"):
    check_tracking_server(tracking_uri)
    model_config = ModelConfig()
    data_config = DataConfig()
    model_config.vocab_size = get_vocab_size(tokenizer_type, data_config)
    pad_idx = get_pad_idx(tokenizer_type, data_config)

    results_dir = os.path.join(RESULTS_ROOT, tokenizer_type)
    os.makedirs(results_dir, exist_ok=True)

    np.random.seed(1337)
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    train_loader = load_data(model_config, data_config, 'train', pad_idx, tokenizer_type)
    val_loader = load_data(model_config, data_config, 'val', pad_idx, tokenizer_type)
    iters_per_epoch = len(train_loader)
    total_iters = model_config.n_epochs * iters_per_epoch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{tokenizer_type}] device={device} vocab_size={model_config.vocab_size} pad_idx={pad_idx}")

    torch.backends.cuda.matmul.allow_tf32 = True

    model = Transformer_Decoder(model_config)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=model_config.learning_rate)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Recipe Transformer Decoder")
    experiment = mlflow.get_experiment_by_name("Recipe Transformer Decoder")
    experiment_id = experiment.experiment_id
    print(f"experiment_id: {experiment_id}")

    history = {"train_loss": [], "val_loss": [], "train_ppl": [], "val_ppl": []}

    with mlflow.start_run(run_name=f"decoder-{tokenizer_type}") as run:
        mlflow.set_tag("tokenizer_type", tokenizer_type)
        params = dict(model_config.__dict__)
        params["optimizer"] = optimizer.__class__.__name__
        params["tokenizer_type"] = tokenizer_type
        mlflow.log_params(params)

        model_summary_path = os.path.join(results_dir, "model_summary.txt")
        with open(model_summary_path, "w") as f:
            f.write(str(summary(model)))
        mlflow.log_artifact(model_summary_path)

        best_val_loss = float("inf")
        best_epoch = None
        best_state = None
        model_name = None
        for i in range(model_config.n_epochs):
            train_loss, train_ppl = train(model, train_loader, optimizer, model_config, i + 1, device, pad_idx, iters_per_epoch, total_iters)
            val_loss, val_ppl = evaluate(model, val_loader, i + 1, device, pad_idx)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_ppl"].append(train_ppl)
            history["val_ppl"].append(val_ppl)

            model_name = f"model{i}"
            mlflow.pytorch.log_model(model, name=model_name)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = i + 1
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        mlflow.set_tag("best_epoch", best_epoch)
        mlflow.log_metric("best_val_loss", best_val_loss)
        print(f"[{tokenizer_type}] best epoch: {best_epoch} (val_loss={best_val_loss:.4f}); "
              f"restoring its weights for final results/generation")
        model.load_state_dict(best_state)
        history["best_epoch"] = best_epoch

        with open(os.path.join(results_dir, "metrics.json"), "w") as f:
            json.dump(history, f, indent=2)

        loss_fig = loss_plot(history)
        loss_fig.savefig(os.path.join(results_dir, "loss_curve.png"), dpi=150)
        mlflow.log_figure(loss_fig, "loss_curve.png")

        ppl_fig = ppl_plot(history)
        ppl_fig.savefig(os.path.join(results_dir, "perplexity_curve.png"), dpi=150)
        mlflow.log_figure(ppl_fig, "perplexity_curve.png")

        gen_tokenizer = Tokenizer(tokenizer_type=tokenizer_type)
        prompts = ["chicken, rice, curry", "flour, sugar, butter, eggs", "tomato, basil, mozzarella"]
        with open(os.path.join(results_dir, "sample_generations.txt"), "w") as f:
            for prompt in prompts:
                samples = generate(model, gen_tokenizer, pad_idx, device, prompt, num_return_sequences=2, max_length=124)
                for sample in samples:
                    f.write(sample + "\n\n")
        mlflow.log_artifact(os.path.join(results_dir, "sample_generations.txt"))

        with open(os.path.join(results_dir, "latest_run.json"), "w") as f:
            json.dump({
                "run_id": run.info.run_id,
                "experiment_id": experiment_id,
                "model_name": f"model{best_epoch - 1}",
                "tracking_uri": tracking_uri,
                "tokenizer_type": tokenizer_type,
                "best_epoch": best_epoch,
            }, f, indent=2)

    del train_loader, val_loader, model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the recipe decoder for one or all tokenizers.")
    parser.add_argument(
        "--tokenizer", choices=TOKENIZER_TYPES, default=None,
        help="Tokenizer to train with. Omit to train one decoder per tokenizer (custom, hug, tiktoken) sequentially.",
    )
    parser.add_argument("--tracking-uri", default="http://localhost:5000")
    args = parser.parse_args()

    tokenizer_types = [args.tokenizer] if args.tokenizer else list(TOKENIZER_TYPES)
    for tokenizer_type in tokenizer_types:
        print(f"\n=== Training decoder for tokenizer_type={tokenizer_type} ===")
        run_training(tokenizer_type, tracking_uri=args.tracking_uri)
