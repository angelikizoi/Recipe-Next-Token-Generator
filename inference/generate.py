import contextlib
import json
import os
import mlflow
import torch
import torch.nn.functional as F


from tokenizer.tokenizer import Tokenizer, get_pad_idx
from config.data_config import DataConfig
from config.inference_config import InferenceConfig

RESULTS_ROOT = "results"


def generate(model, tokenizer, pad_idx, device, prompt, num_return_sequences=4, max_length=None, top_k=50, seed=1337, skip_special_tokens=True):
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
    if max_length is None:
        max_length = context_window
    else:
        max_length = min(max_length, context_window)
    eos_id = tokenizer.eos_id

    prompt_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long)
    generated_ids = prompt_ids.unsqueeze(0).repeat(num_return_sequences, 1).to(device)
    finished = torch.zeros(num_return_sequences, dtype=torch.bool, device=device)

    # bf16 autocast only on the GPU; on the CPU generation runs in plain fp32
    if device.type == "cuda":
        autocast = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        autocast = contextlib.nullcontext()

    with torch.no_grad():
        while generated_ids.size(1) < max_length and not finished.all():
            with autocast:
                logits, _ = model(pad_idx, generated_ids)  # num_return_sequences, seq_length, vocab_size
            logits = logits[:, -1, :]  # num_return_sequences, vocab_size
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_ids = torch.topk(probs, top_k, dim=-1)
            topk_choice = torch.multinomial(topk_probs, 1)
            next_ids = torch.gather(topk_ids, -1, topk_choice)  # num_return_sequences, 1
            # sequences that already stopped keep emitting pad, so the batch stays aligned
            next_ids[finished] = pad_idx
            generated_ids = torch.cat((generated_ids, next_ids), dim=1)
            finished |= next_ids.squeeze(1) == eos_id

    samples = []
    for i in range(num_return_sequences):
        text_ids = generated_ids[i].tolist()
        if eos_id in text_ids:
            text_ids = text_ids[:text_ids.index(eos_id) + 1]
        samples.append(tokenizer.decode(text_ids, skip_special_tokens=skip_special_tokens))
    return samples


def load_model_from_mlflow(run_id, model_name, tracking_uri):
    mlflow.set_tracking_uri(tracking_uri)
    return mlflow.pytorch.load_model(f"runs:/{run_id}/{model_name}")


def resolve_run(config):
    """Fall back to results/<tokenizer>/latest_run.json (written by training/train.py) when no run_id is set."""
    if config.run_id:
        return config.run_id, config.model_name
    latest_run_path = os.path.join(RESULTS_ROOT, config.tokenizer_type, "latest_run.json")
    if not os.path.exists(latest_run_path):
        raise SystemExit(
            f"No run_id set and {latest_run_path} not found. "
            "Run training/train.py first, or set run_id / model_name in config/inference_config.py."
        )
    with open(latest_run_path) as f:
        latest = json.load(f)
    return latest["run_id"], config.model_name or latest["model_name"]


if __name__ == "__main__":
    config = InferenceConfig()

    run_id, model_name = resolve_run(config)
    pad_idx = get_pad_idx(config.tokenizer_type, DataConfig())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from run {run_id}/{model_name} ({config.tracking_uri})...")
    model = load_model_from_mlflow(run_id, model_name, config.tracking_uri).to(device)
    tokenizer = Tokenizer(tokenizer_type=config.tokenizer_type)

    samples = generate(model, tokenizer, pad_idx, device, config.prompt, num_return_sequences=config.num_return_sequences,
                       max_length=config.max_length, skip_special_tokens=config.skip_special_tokens)
    for i, sample in enumerate(samples):
        print(f"sample {i}: {sample}")