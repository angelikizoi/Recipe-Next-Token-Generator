# 🍽️ Recipe Next-Token Generator

### *Decoder-Only Transformer (GPT-2 Style) Trained on Recipe Data*

This project implements a **GPT-2–inspired decoder-only Transformer** for **next-token prediction** on cooking recipes.
It includes a **custom BPE tokenizer**, **HuggingFace tokenizer compatibility**, **MLflow experiment tracking**, and a modular training pipeline.

---

## 📌 Features

### 🔥 Transformer Model (GPT-2 Style)

* Decoder-only architecture
* Causal self-attention
* Configurable hyperparameters via `model_config.py`
* Supports both **training** and **inference**

### 🧠 Custom Tokenizer (BPE)

* Built from scratch with:

  * Multiprocessing
  * Generator-based streaming
* Trained on the Recipe 2M dataset (https://www.kaggle.com/datasets/wilmerarltstrmberg/recipe-dataset-over-2m)
* HuggingFace tokenizer conversion included
* Stored in `tokenizer_train/custom_tokenizer_files/`

### 🍲 Dataset: Kaggle Recipe 2M

* Includes ingredients + instructions
* Preprocessing scripts located in `dataset/data_preprocessor.py`
* Cleaned output accessible via:

  * `dataset/full_dataset.csv`
  * `dataset/recipe.txt`
  * etc.

### 📊 MLflow Experiment Tracking

* Logs:

  * Loss curves
  * Model configs
  * Tokenizer metadata
  * Checkpoints & artifacts

### 🧱 Modular Project Architecture

As in the screenshot:

```
NLP/
│
├── config/
│   ├── data_config.py
│   ├── model_config.py
│
├── dataloader/
│   ├── dataloader.py
│
├── dataset/
│   ├── data_preprocessor.py
│   ├── dataset.zip
│   ├── full_dataset.csv
│   └── ...
│
├── tokenizer_train/
│   ├── custom_tokenizer_files/
│   ├── hugface_tokenizer.py
│   ├── custom_tokenizer_train.py
│
├── inference/
│   ├── inference.py
│
├── transformer.py
├── transformer_train.py
├── play.ipynb
├── model_summary.txt
└── README.md
```

---

## 🛠️ Future Improvements

* Further Improvement of GPT Architecture & Custom Tokenizer
* Recipe generation web UI - Streamlit Implementation


---

