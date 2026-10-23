from dataclasses import dataclass

@dataclass
class InferenceConfig:
    prompt: str = "chicken, rice, curry"
    tokenizer_type: str = "hug"  # the tokenizer the checkpoint was trained with: "custom" or "hug"
    run_id: str | None = None  # None -> the run in results/<tokenizer_type>/latest_run.json
    model_name: str | None = None  # MLflow logged-model name, e.g. "model9"; None -> the one in latest_run.json
    tracking_uri: str = "http://localhost:5000"
    num_return_sequences: int = 4
    max_length: int | None = None  # safety cap; None -> the model's context window (generation stops at <|EOS|>)
    skip_special_tokens: bool = True  # False keeps the <|TITLE|>/<|INGREDIENTS|>/<|DIRECTIONS|>/<|EOS|> markers