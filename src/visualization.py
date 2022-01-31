import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from typing import List, Optional
from numpy import ndarray
from pathlib import Path


def visualize_attrs(attrs: ndarray, token_words: List[str], save_str: Optional[str] = None):
    """makes bar charts that show how important each token-word is"""
    assert len(attrs) == len(token_words)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    fake_x = range(len(token_words))

    ax.bar(fake_x, attrs)
    ax.set_xticks(fake_x)
    # ax.set_yticks(token_words)
    ax.set_xticklabels(token_words)
    # ax.set_ylabel("F1", fontsize=16)

    plt.tight_layout()
    if save_str is not None:
        if not save_str.endswith(".png"):
            save_str += ".png"
        plt.savefig(Path("figures") / Path(save_str))
    else:
        plt.show()


def embedding_histogram(embeddings: Tensor):
    flattened = torch.flatten(embeddings).detach().cpu().numpy()
    print(f"{np.min(flattened)=}")
    print(f"{np.max(flattened)=}")
    print(f"{np.mean(flattened)=}")
    print(f"{np.std(flattened)=}")
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.hist(flattened, bins=1000)
    plt.tight_layout()
    plt.show()
