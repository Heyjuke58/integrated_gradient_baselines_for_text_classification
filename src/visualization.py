import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
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
    model_str: str,
    version_ig: str,
    sentence: str,
    token_words: List[str],
    save_str: Optional[str] = None,
) -> None:
    """makes bar charts that show how important each token-word is"""
    num_bbs = len(bl_attrs)
    fig, axs = plt.subplots(
        num_bbs, 1, sharex=True, squeeze=False, figsize=(16, 9), gridspec_kw={"hspace": 0}
    )
    axins1 = inset_axes(
        axs[0, 0],
        width="100%",
        height="100%",
        # loc="upper right",
        bbox_to_anchor=(0.78, 1.4, 0.2, 0.1),
        bbox_transform=axs[0, 0].transAxes,
    )
    x_indices = range(len(token_words))
    pal = sns.color_palette("summer", len(token_words))

    # loop over different input sentences:
    for i, (bl_name, attrs) in enumerate(bl_attrs.items()):
        df = pd.DataFrame(attrs, columns=["attr"])
        rank = df.rank(axis=0, method="min", ascending=False)["attr"].to_numpy(dtype=np.int8)
        sns.barplot(
            x=df.index,
            y="attr",
            data=df,
            palette=np.array(pal)[rank],
            # hue="attr",
            ax=axs[i, 0],
            orient="v",
        )
        axs[i, 0].set_xticks(x_indices)
        # ax.set_yticks(token_words)
        axs[i, 0].set_xticklabels(token_words, rotation=30, ha="right")
        axs[i, 0].set_ylabel(bl_name, rotation=0, ha="right")

    # axs[0, 0].legend()
    cbar = fig.colorbar(
        cm.ScalarMappable(norm=None, cmap="summer"),
        cax=axins1,
        orientation="horizontal",
        ticks=[0, 1],
    )
    cbar.ax.set_xticklabels(["high rank", "low rank"])
    fig.text(
        0.5,
        0.95,
        f"{version_ig.upper()} attributions for {model_str.upper()} model\n(Sum of absolute cumulative gradients)",
        ha="center",
        fontsize=16,
    )
    plt.suptitle(f"sentence: {sentence}\nPrediction: {prediction}", y=0.05)
    # plt.tight_layout()
    if save_str is not None:
        if not save_str.endswith(".png"):
            save_str += ".png"
        plt.savefig(Path("figures") / Path(save_str))
    else:
        plt.show()


def embedding_histogram(embeddings: Tensor) -> None:
    flattened = torch.flatten(embeddings).detach().cpu().numpy()
    mu = np.mean(flattened)
    sigma = np.std(flattened)
    top = mu + 2.58 * sigma
    bot = mu - 2.58 * sigma
    count = np.sum((flattened > bot) & (flattened < top))

    flattened_sorted = np.sort(flattened)
    n_outliers = round(len(flattened) * 0.001)
    soft_top = flattened_sorted[n_outliers]
    soft_bot = flattened_sorted[-n_outliers]

    print(f"{np.min(flattened)=}")
    print(f"{np.max(flattened)=}")
    print(f"{mu=}")
    print(f"{sigma=}")
    print(f"{top=}")
    print(f"{bot=}")
    print(f"{soft_top=}")
    print(f"{soft_bot=}")
    print(f"count inside inteval: {count} ({(100 * count) / len(flattened):.2f}%)")
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.axvline(mu, color="black")
    ax.axvline(soft_top, color="green")
    ax.axvline(soft_bot, color="green")
    ax.axvline(top, color="red")
    ax.axvline(bot, color="red")
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
