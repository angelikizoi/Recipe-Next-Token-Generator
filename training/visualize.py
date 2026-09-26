import matplotlib.pyplot as plt
import numpy as np



def loss_plot(history):
    epochs = np.arange(1, len(history['train_loss']) + 1)
    with plt.style.context(style='seaborn-v0_8-pastel'):
        fig, ax = plt.subplots(figsize=(10,8))
        ax.plot(epochs, history["train_loss"], marker="o", label="Train loss")
        ax.plot(epochs, history["val_loss"], marker="o", label="Val loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-entropy loss")
        ax.set_title("Train/Val Loss")
        ax.legend()
        plt.grid(True)
        plt.tight_layout()
    plt.close(fig)
    return fig

def ppl_plot(history):
    epochs = np.arange(1, len(history['train_ppl']) + 1)
    with plt.style.context(style='seaborn-v0_8-pastel'):
        fig, ax = plt.subplots(figsize=(10,8))
        ax.plot(epochs, history["train_ppl"], marker="o", label="Train PPL")
        ax.plot(epochs, history["val_ppl"], marker="o", label="Val PPL")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Perplexity")
        ax.set_title("Train/Val Perplexity")
        ax.legend()
        plt.grid(True)
        plt.tight_layout()
    plt.close(fig)
    return fig
