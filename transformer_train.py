import math
import time
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
import torch
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm


from transformer import Transformer_Decoder
from dataloader.dataloader import TokenizedDataset, SentenceLenSampler, collate_pad_fn
from config.model_config import ModelConfig
from config.data_config import DataConfig


def load_data(model_config, data_config, mode, pad_idx):
    line_offsets = np.load(data_config.offsets_filepath)
    lengths = (line_offsets[1:] - line_offsets[:-1]).astype(np.int64)
    n = len(lengths)
    indices = np.random.permutation(n)
    if mode == 'train':
        sampler_indices = indices[:int(model_config.train_test_split*n)]
    else:
        sampler_indices = indices[int(model_config.train_test_split*n):]

    data = TokenizedDataset(line_offsets, model_config.sentence_length, tokenizer_type='hug')
    return DataLoader(
        data,
        batch_sampler=SentenceLenSampler(sampler_indices, lengths, model_config.batch_size),
        collate_fn=lambda b: collate_pad_fn(b, pad_token_id=pad_idx),
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )

def train(model, train_loader, optimizer, model_config, epoch, device, pad_idx):
    model.train()
    t0 = time.time()
    train_loss = 0
    train_cnt = 0
    train_iterator = tqdm(train_loader, desc=f"Training Epoch {epoch:02d}")
    train_total_batches = len(train_loader)
    for batch_step, batch in enumerate(train_iterator):
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
            mlflow.log_metric("loss", loss.item(), step=((batch_step // 200) + (epoch - 1) * (train_total_batches//200 + 1)))
    torch.cuda.synchronize()
    t1 = time.time()
    train_dt = (t1 - t0) / 60
    avg_train_loss = train_loss / train_cnt
    mlflow.log_metric("train_dt", train_dt, step=epoch)
    mlflow.log_metric("avg_train_loss", avg_train_loss, step=epoch)
    mlflow.log_metric("train_ppl", math.exp(avg_train_loss), step=epoch)
    mlflow.log_metric("norm", norm.item(), step=epoch)
    print(f"epoch: {epoch}, train_loss: {avg_train_loss:.4f}, training_time: {train_dt}mins")

def eval(model, val_loader, epoch, device, pad_idx):
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


if __name__ == "__main__":
    model_config = ModelConfig()
    data_config = DataConfig()
    pad_idx = data_config.special_tokens["<|PAD|>"]
    
    train_loader = load_data(model_config, data_config, 'train', pad_idx)
    val_loader = load_data(model_config, data_config, 'val', pad_idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{device=}")

    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    torch.backends.cuda.matmul.allow_tf32 = True

    model = Transformer_Decoder(model_config)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=model_config.learning_rate)

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Recipe Transformer Decoder")
    experiment = mlflow.get_experiment_by_name("Recipe Transformer Decoder")
    experiment_id = experiment.experiment_id
    print(f"experiment_id: {experiment_id}")

    client = MlflowClient() 
    mlflow.set_experiment(experiment_id=experiment_id)

    with mlflow.start_run():
        model_config = ModelConfig()
        params = model_config.__dict__
        params["optimizer"] = optimizer.__class__.__name__,
        mlflow.log_params(params)

        with open("model_summary.txt", "w") as f:
            f.write(str(summary(model)))
        mlflow.log_artifact("model_summary.txt")

        for i in range(model_config.n_epochs):
            train(model, train_loader, optimizer, model_config, i + 1, device, pad_idx)
            eval(model, val_loader, i + 1, device, pad_idx)

            mlflow.pytorch.log_model(model, name=f"model{i}")

