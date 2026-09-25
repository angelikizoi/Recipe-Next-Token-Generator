import mlflow
import torch
import torch.nn.functional as F

from dataloader.tokenizer import Tokenizer
from config.data_config import DataConfig

pad_idx = DataConfig().special_tokens["<|PAD|>"]

mlflow.set_tracking_uri('http://localhost:5000')
run_id = "afbfd4c68e5d4fcd8fd37e4e0da672f7"
logged_model_path = f"runs:/{run_id}/model8"
loaded_model = mlflow.pytorch.load_model(logged_model_path)

tokenizer = Tokenizer(tokenizer_type='hug')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

num_return_sequences = 4
max_length = 124
tokens = tokenizer.encode("chicken, rice, curry")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
xgen = tokens.to(device)
loaded_model.eval()
while xgen.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, loss = loaded_model(pad_idx, xgen) # (B, T, vocab_size)
        # take the logits at the last position
        logits = logits[:, -1, :] # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probabilities
        # note: multinomial does not demand the input to sum to 1
        ix = torch.multinomial(topk_probs, 1) # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        # append to the sequence
        xgen = torch.cat((xgen, xcol), dim=1)
# print the generated text
for i in range(num_return_sequences):
    tokens = xgen[i, :max_length].tolist()
    decoded = tokenizer.decode(tokens)
    print(f"sample {i}: {decoded}")