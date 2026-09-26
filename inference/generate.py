import argparse
import contextlib
import json
import os

import mlflow
import torch
import torch.nn.functional as F

from tokenizer.tokenizer import Tokenizer, get_pad_idx
from config.data_config import DataConfig

RESULTS_ROOT = "results"


@torch.no_grad()
def generate(model, tokenizer, pad_idx, device, prompt, num_return_sequences=4, max_length=None,
             top_k=50, seed=1337, skip_special_tokens=True):
    """Top-k sampling from `model`, starting from `prompt`.

    Each sequence runs until it emits <|EOS|>. `max_length` is only a safety net and
    defaults to the model's context window, which the forward pass cannot exceed.
    Returns a list of decoded strings.
    """
    model.eval()
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)

    context_window = model.config.sentence_length
    max_length = context_window if max_length is None else min(max_length, context_window)
    eos_id = tokenizer.eos_id

    tokens = torch.tensor(tokenizer.encode(prompt), dtype=torch.long)
    xgen = tokens.unsqueeze(0).repeat(num_return_sequences, 1).to(device)
    finished = torch.zeros(num_return_sequences, dtype=torch.bool, device=device)

    # CPU autocast supports only bf16/fp16, so plain fp32 is used off-GPU
    autocast = (torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device.type == "cuda" else contextlib.nullcontext())
    while xgen.size(1) < max_length and not bool(finished.all()):
        with autocast:
            logits, _ = model(pad_idx, xgen)  # (B, T, vocab_size)
        logits = logits[:, -1, :]  # (B, vocab_size)
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
        ix = torch.multinomial(topk_probs, 1)  # (B, 1)
        xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
        # sequences that already stopped keep emitting pad, so the batch stays aligned
        xcol = torch.where(finished.unsqueeze(1), torch.full_like(xcol, pad_idx), xcol)
        xgen = torch.cat((xgen, xcol), dim=1)
        finished |= xcol.squeeze(1) == eos_id

    samples = []
    for i in range(num_return_sequences):
        ids = xgen[i].tolist()
        if eos_id in ids:
            ids = ids[: ids.index(eos_id) + 1]
        samples.append(tokenizer.decode(ids, skip_special_tokens=skip_special_tokens))
    return samples


def load_model_from_mlflow(run_id, model_name, tracking_uri):
    mlflow.set_tracking_uri(tracking_uri)
    return mlflow.pytorch.load_model(f"runs:/{run_id}/{model_name}")


def _resolve_run(args):
    """Fall back to results/<tokenizer>/latest_run.json (written by training/train.py) when --run-id is omitted."""
    if args.run_id:
        return args.run_id, args.model_name
    latest_run_path = os.path.join(RESULTS_ROOT, args.tokenizer, "latest_run.json")
    if not os.path.exists(latest_run_path):
        raise SystemExit(
            f"No --run-id given and {latest_run_path} not found. "
            "Run training/train.py first, or pass --run-id / --model-name explicitly."
        )
    with open(latest_run_path) as f:
        latest = json.load(f)
    return latest["run_id"], args.model_name or latest["model_name"]


def main():
    parser = argparse.ArgumentParser(description="Generate recipe text from a trained checkpoint.")
    parser.add_argument("--prompt", default="chicken, rice, curry", help="Prompt to condition generation on.")
    parser.add_argument("--tokenizer", choices=["custom", "hug", "tiktoken"], default="hug",
                         help="Tokenizer the target checkpoint was trained with.")
    parser.add_argument("--run-id", default=None, help="MLflow run id. Defaults to results/<tokenizer>/latest_run.json.")
    parser.add_argument("--model-name", default=None, help="MLflow logged-model name, e.g. 'model9'.")
    parser.add_argument("--tracking-uri", default="http://localhost:5000")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=None,
                        help="Safety cap on generated length. Defaults to the model's context window; "
                             "generation normally stops at <|EOS|>.")
    parser.add_argument("--show-special-tokens", action="store_true",
                        help="Render <|TITLE|>/<|INGREDIENTS|>/<|DIRECTIONS|>/<|EOS|> markers in the output.")
    args = parser.parse_args()

    run_id, model_name = _resolve_run(args)
    pad_idx = get_pad_idx(args.tokenizer, DataConfig())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from run {run_id}/{model_name} ({args.tracking_uri})...")
    model = load_model_from_mlflow(run_id, model_name, args.tracking_uri).to(device)
    tokenizer = Tokenizer(tokenizer_type=args.tokenizer)

    samples = generate(
        model, tokenizer, pad_idx, device, args.prompt,
        num_return_sequences=args.num_samples, max_length=args.max_length,
        skip_special_tokens=not args.show_special_tokens,
    )
    for i, sample in enumerate(samples):
        print(f"sample {i}: {sample}")


if __name__ == "__main__":
    main()
