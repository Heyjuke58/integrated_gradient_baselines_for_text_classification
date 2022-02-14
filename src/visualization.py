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


# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc("axes", titlesize=16)  # fontsize of the axes title
plt.rc("axes", labelsize=14)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=16)  # fontsize of the tick labels
plt.rc("ytick", labelsize=12)  # fontsize of the tick labels
plt.rc("legend", fontsize=14)  # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def visualize_attrs(
    bl_attrs: Dict[str, ndarray],
    prediction: str,
    true: str,
    model_str: str,
    version_ig: str,
    sentence: str,
    token_words: List[str],
    save_str: Optional[str] = None,
) -> None:
    """makes bar charts that show how important each token-word is"""
    num_bbs = len(bl_attrs)
    fig, axs = plt.subplots(
        num_bbs,
        1,
        sharex=True,
        squeeze=False,
        figsize=(19.20, 10.80),
        gridspec_kw={"hspace": 0},
    )
    x_indices = range(len(token_words))
    pal = cm.get_cmap("coolwarm")

    # loop over different input sentences:
    for i, (bl_name, attrs) in enumerate(bl_attrs.items()):
        attrs = attrs / np.max(np.abs(attrs))
        df = pd.DataFrame(attrs, columns=["attr"])
        sns.barplot(
            x=df.index,
            y="attr",
            data=df,
            palette=pal((attrs + 1) / 2),
            # hue="attr",
            ax=axs[i, 0],
            orient="v",
            zorder=10,
        )
        axs[i, 0].set_xticks(x_indices)
        axs[i, 0].set_xticklabels(token_words, rotation=30, ha="right")
        axs[i, 0].set_ylabel(bl_name, rotation=30, ha="right")

        # set background color
        axs[i, 0].axhspan(facecolor="#DADADA", alpha=0.4, ymin=-1, ymax=1, zorder=-1)

        # hide frame
        axs[i, 0].spines["top"].set_visible(False)
        axs[i, 0].spines["bottom"].set_visible(False)

        # unify y axis
        axs[i, 0].set_ylim([-1.1, 1.1])
        axs[i, 0].set_yticks([0])

        # x axis line:
        axs[i, 0].axhline(0, color="black", linewidth=0.8, zorder=11)

    axs[0, 0].spines["top"].set_visible(True)
    axs[-1, 0].spines["bottom"].set_visible(True)

    fig.text(
        0.5,
        0.92,
        f"{version_ig.upper()} attributions for {model_str.upper()} model\n(Sum of cumulative gradients)",
        ha="center",
        fontsize=24,
    )
    plt.suptitle(f"True: {true} Prediction: {prediction}", y=0.03, fontsize=20)
    # plt.tight_layout()
    if save_str is not None:
        if not save_str.endswith(".png"):
            save_str += ".png"
        plt.savefig(Path("figures/attributions") / Path(save_str), bbox_inches="tight")
    else:
        plt.show()


def embedding_histogram(embeddings: Tensor) -> None:
    """
    Plots the distribution of values in the embeddings of the whole vocab of a model
    """
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
    """
    Plots TopK ablations scores (either comprehensiveness or log odds) of explanations for a model
    """
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(19.20, 10.80), gridspec_kw={"hspace": 0})

    # set cyclic color map
    cm = plt.get_cmap("gist_rainbow")
    NUM_COLORS = len(list(avg_scores.values()))
    ax.set_prop_cycle(color=[cm(1.0 * i / NUM_COLORS) for i in range(NUM_COLORS)])

    for i, (bl_name, scores) in enumerate(avg_scores.items()):
        x = list(scores.keys())
        y = list(scores.values())
        ax.plot(x, y, label=bl_name, marker="o")
    ax.set_xticks(list(list(avg_scores.values())[0].keys()))
    ax.set_xlabel("top-k % of tokens masked")
    ax.set_ylabel(ablation_str)
    ax.legend()

    plt.suptitle(
        f"TopK ablation of {num_examples} examples for {model_str} model: {ablation_str}",
        y=0.05,
        fontsize=24,
    )

    if save_str is not None:
        if not save_str.endswith(".png"):
            save_str += ".png"
        plt.savefig(Path("figures/ablation") / Path(save_str), bbox_inches="tight")
    else:
        plt.show()


def visualize_embedding_space(
    word_emb: np.ndarray, pca, interesting_embs: Dict[str, np.ndarray]
) -> None:
    """
    Plots a PCA of the embeddings space of a model. Additional interesting embeddings can be passed.
    """
    fig, ax = plt.subplots(1, 1, figsize=(19.20, 10.80), gridspec_kw={"hspace": 0})
    word_emb = pca.transform(word_emb)
    ax.scatter(word_emb[:, 0], word_emb[:, 1], alpha=0.2, marker=".", edgecolors=None, linewidths=0)

    for word, emb in interesting_embs.items():
        emb = pca.transform(np.expand_dims(emb, axis=0))
        ax.annotate(word, (emb[:, 0], emb[:, 1]))

    ax.set_xlabel("1st PCA dimension")
    ax.set_ylabel("2nd PCA dimension")
    plt.title("Different words in the embedding space (2-dimensional PCA)")
    plt.savefig("figures/pca_embedding_space.png")
    plt.show()


def visualize_word_paths(
    word_path_emb: np.ndarray,  # (w, l, 768)
    word_path_discretized_emb: np.ndarray,  # (w, l, 768)
    word_path: List[List[str]],  # [["PAD", ..., "good"], ["PAD",... "movie"]]
    cloud_emb: np.ndarray,  # embs of full vocabulary
    pca,
    model_str: str,
    version_ig: str,
    save_str: Optional[str] = None,
) -> None:
    """
    Visualizes closest by tokens of interpolated word paths of a sentence with a PCA
    """
    fig, ax = plt.subplots(1, 1, figsize=(19.20, 10.80), gridspec_kw={"hspace": 0})

    # make vocabulary cloud in background:
    cloud_pca = pca.transform(cloud_emb)
    ax.scatter(
        cloud_pca[:, 0],
        cloud_pca[:, 1],
        color="lightgray",
        marker=".",
        edgecolors=None,
        linewidths=0,
    )

    # visualize word paths:
    cmap = cm.get_cmap("tab10")
    for p, (path, disc_path, words) in enumerate(
        zip(word_path_emb, word_path_discretized_emb, word_path)
    ):
        path_pca = pca.transform(path)
        disc_path_pca = pca.transform(disc_path)
        ax.plot(
            path_pca[:, 0],
            path_pca[:, 1],
            color=cmap(p),
            marker="o",
            linewidth=1.5,
            label=f"{words[-1]}: Actual interpolation",
        )
        ax.plot(
            disc_path_pca[:, 0],
            disc_path_pca[:, 1],
            color=cmap(p),
            ls="dotted",
            marker="o",
            linewidth=2,
            label=f"{words[-1]}: Discretized interpolation",
        )
        last_word = None
        for i, word in enumerate(words):
            if word != last_word:
                ax.annotate(word, (disc_path_pca[i, 0], disc_path_pca[i, 1]), fontsize=14)
                last_word = word
    plt.legend()
    plt.title(
        f"Interpolation path of the sentence to a baseline and decoding to the closest-by tokens (2-dimensional PCA)\n{version_ig.upper()} with {model_str.upper()} model",
        fontsize=24,
    )
    ax.set_xlabel("1st PCA dimension")
    ax.set_ylabel("2nd PCA dimension")

    # set ranges for better visualization of certain examples
    ax.set_xlim([-0.3, 0.9])
    ax.set_ylim([-0.5, 0.25])

    plt.tight_layout()
    if save_str is not None:
        if not save_str.endswith(".png"):
            save_str += ".png"
        plt.savefig(Path("figures/wordpath") / Path(save_str), bbox_inches="tight")
    else:
        plt.show()


def visualize_word_path(
    word_path_emb: np.ndarray,
    word_path_discretized_emb: np.ndarray,
    word_path: List[str],
    pca,
    model_str: str,
    version_ig: str,
    save_str: Optional[str] = None,
) -> None:
    """
    Visualizes closest by tokens of interpolated word paths of one word of a sentence with a PCA
    """
    fig, ax = plt.subplots(1, 1, figsize=(19.20, 10.80), gridspec_kw={"hspace": 0})
    word_path_emb = pca.transform(word_path_emb)
    word_path_discretized_emb = pca.transform(word_path_discretized_emb)
    ax.plot(word_path_emb[:, 0], word_path_emb[:, 1], marker="o", label="Actual interpolation")
    ax.plot(
        word_path_discretized_emb[:, 0],
        word_path_discretized_emb[:, 1],
        marker="o",
        label="Discretized interpolation",
    )
    plt.legend()
    plt.title(
        f"Interpolation path of a word to a baseline and decoding to the closest-by tokens (2-dimensional PCA)\n{version_ig.upper()} with {model_str.upper()} model",
        fontsize=24,
    )
    ax.set_xlabel("1st PCA dimension")
    ax.set_ylabel("2nd PCA dimension")
    last_word = None
    for i, word in enumerate(word_path):
        if word != last_word:
            ax.annotate(word, (word_path_discretized_emb[i, 0], word_path_discretized_emb[i, 1]))
            last_word = word
    if save_str is not None:
        if not save_str.endswith(".png"):
            save_str += ".png"
        plt.savefig(Path("figures/wordpath") / Path(save_str), bbox_inches="tight")
    else:
        plt.show()


def visualize_word_path_table(
    word_path: List[List[str]],  # [["PAD", ..., "good"], ["PAD",... "movie"]]
    model_str: str,
    version_ig: str,
    baseline_str: str,
    save_str: Optional[str] = None,
) -> None:
    """
    Visualizes closest by tokens of interpolated word paths of a sentence in a table
    """
    description = [""] * len(word_path[0])
    description[0] = "BASELINE"
    description[-1] = "ORIGINAL"
    word_path = [description] + word_path
    np_word_path = np.asarray(word_path).T[::-1, :]
    fig, ax = plt.subplots(figsize=(19.20, 10.80))
    ax.axis("off")
    table = ax.table(
        cellText=np_word_path,
        loc="center",
        edges="open",
        cellLoc="center",
        bbox=[0.1, 0.1, 0.8, 0.8],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(24)
    table[(0, 0)].get_text().set_color("red")
    table[(len(word_path[0]) - 1), 0].get_text().set_color("red")
    plt.title(
        f"Closest by tokens of interpolated paths.\n{version_ig.upper()} for {model_str.upper()} (Baseline: {baseline_str})",
        fontsize=24,
    )

    if save_str is not None:
        if not save_str.endswith(".png"):
            save_str += ".png"
        plt.savefig(Path("figures/wordpath_table") / Path(save_str), bbox_inches="tight")
    else:
        plt.show()
