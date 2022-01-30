import matplotlib.pyplot as plt
from typing import List
from numpy import ndarray

def visualize_attrs(attrs: ndarray, token_words: List[str]):
    """makes bar charts that show how important each token-word is"""
    assert len(attrs) == len(token_words)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    fake_x = range(len(token_words))

    ax.bar(fake_x, attrs)
    ax.set_xticks(fake_x)
    # ax.set_yticks(token_words)
    ax.set_xticklabels(token_words)
    # ax.set_ylabel("F1", fontsize=16)
    
    plt.show()