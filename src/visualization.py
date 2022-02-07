import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import torch
from torch import Tensor
from typing import List, Optional, Dict
from numpy import ndarray
from pathlib import Path
import seaborn as sns
import pandas as pd


def visualize_attrs(
    bl_attrs: Dict[str, ndarray],
    prediction: str,
    true_label: str,
    model_str: str,
    version_ig: str,
    sentence: str,
    token_words: List[str],
    save_str: Optional[str] = None,
) -> None:
    """makes bar charts that show how important each token-word is"""
    num_bbs = len(bl_attrs)
    fig, axs = plt.subplots(
        num_bbs, 1, sharex=True, squeeze=False, figsize=(16, 8), gridspec_kw={"hspace": 0}
    )
    x_indices = range(len(token_words))
    pal = sns.color_palette("bwr", len(token_words))

    # loop over different input sentences:
    # for j in range(bl_attrs.values()[0].shape[0]):
    for i, (bl_name, attrs) in enumerate(bl_attrs.items()):
        df = pd.DataFrame(attrs, columns=["attr"])
        rank = df.rank(axis=0)
        sns.barplot(
            x=df.index,
            y="attr",
            data=df,
            palette=np.array(pal[::-1])[rank],
            # hue="attr",
            ax=axs[i, 0],
            orient="v",
        )
        axs[i, 0].set_xticks(x_indices)
        # ax.set_yticks(token_words)
        axs[i, 0].set_xticklabels(token_words)
        axs[i, 0].set_ylabel(bl_name, rotation=0, ha="right")
        # axs[i, 0].legend([], [], frameon=False)

    # axs[0, 0].set_title(f"{version_ig} for {model_str}")
    fig.text(
        0.5, 0.95, f"{version_ig} attributions for {model_str} model", ha="center", fontsize=16
    )
    plt.suptitle(f"sentence: {sentence}\nTrue: {true_label}, Prediction: {prediction}", y=0.05)
    # plt.tight_layout()
    if save_str is not None:
        if not save_str.endswith(".png"):
            save_str += ".png"
        plt.savefig(Path("figures") / Path(save_str))
    else:
        plt.show()


def embedding_histogram(embeddings: Tensor) -> None:
    flattened = torch.flatten(embeddings).detach().cpu().numpy()
    print(f"{np.min(flattened)=}")
    print(f"{np.max(flattened)=}")
    print(f"{np.mean(flattened)=}")
    print(f"{np.std(flattened)=}")
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.hist(flattened, bins=1000)
    plt.tight_layout()
    plt.show()


def visualize_ablation_scores(
    avg_scores: Dict[str, Dict[float, float]],
    model_str: str,
    ablation_str: str,
    num_examples: int,
    save_str: Optional[str] = None,
) -> None:
    num_bbs = len(list(avg_scores.keys()))
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(16, 8), gridspec_kw={"hspace": 0})
    for i, (bl_name, scores) in enumerate(avg_scores.items()):
        x = list(scores.keys())
        y = list(scores.values())
        ax.plot(x, y, label=bl_name)
    ax.set_xticks(list(list(avg_scores.values())[0].keys()))
    ax.set_xlabel("top k of tokens masked")
    ax.set_ylabel(ablation_str)
    ax.legend()

    plt.suptitle(
        f"TopK ablation of {num_examples} examples for {model_str} model: {ablation_str}", y=0.05
    )

    if save_str is not None:
        if not save_str.endswith(".png"):
            save_str += ".png"
        plt.savefig(Path("figures") / Path(save_str))
    else:
        plt.show()
