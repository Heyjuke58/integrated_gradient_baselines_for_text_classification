import torch
from torch import Tensor
from torch.nn.functional import softmax
from src.helper_functions import predict
from typing import Dict, List
from collections import defaultdict
import numpy as np


def calculate_suff(attrs):
    pass


def calculate_comp(
    attr: Tensor,
    k: float,
    replacement_emb: Tensor,
    model,
    input_emb: Tensor,
    attention_mask: Tensor,
    prediction: Tensor,
) -> float:
    """
    Comprehensiveness scoring of an attribution.

    :param attr: Attribution scores for one sentence
    :param k: top-k value (how many embeddings are replaced)
    :param replacement_emb: embedding for one word that should be used as replacement
    :param input_emb: Embedding of the sentence for which the attribution was computed
    :param attention_mask: Original attention mask for the sentence
    :param prediction: what model outputs for the input
    """
    # get logits of masked prediction:
    replaced_embed = replace_k_percent(attr, k, replacement_emb, input_emb)
    new_pred = predict(model, replaced_embed, attention_mask).squeeze(0)

    # convert logits of (original) prediction and new_prediction to probabilities:
    new_pred = softmax(new_pred, dim=0)
    prediction = softmax(prediction, dim=0)

    pred_i = torch.argmax(prediction).item()
    return (new_pred[pred_i] - torch.max(prediction)).item()


def calculate_log_odds(
    attr: Tensor,
    k: float,
    replacement_emb: Tensor,
    model,
    input_emb: Tensor,
    attention_mask: Tensor,
    prediction: Tensor,
) -> float:
    """
    Log-odds scoring of an attribution

    :param attr: Attribution scores for one sentence
    :param k: top-k value (how many embeddings are replaced)
    :param replacement_emb: embedding for one word that should be used as replacement
    :param input_emb: Embedding of the sentence for which the attribution was computed
    :param attention_mask: Original attention mask for the sentence
    :param prediction: what model outputs for the input
    """
    # get logits of masked prediction:
    replaced_embed = replace_k_percent(attr, k, replacement_emb, input_emb)
    new_pred = predict(model, replaced_embed, attention_mask).squeeze(0)

    # convert logits of (original) prediction and new_prediction to probabilities:
    new_pred = softmax(new_pred, dim=0)
    prediction = softmax(prediction, dim=0)

    pred_i = torch.argmax(prediction).item()
    return torch.log(new_pred[pred_i] / torch.max(prediction)).item()


def replace_k_percent(attr: Tensor, k: float, replacement_emb: Tensor, input_emb: Tensor) -> Tensor:
    """
    Given a sentence embedding (without padding tokens at the end) and an attribution scoring over the tokens,
    replace the top-k embeddings with the replacement embedding.
    
    :param attr: Attribution scores for one sentence
    :param k: top-k value (how many embeddings are replaced)
    :param replacement_emb: embedding for one word that should be used as replacement
    :param input_emb: Embedding of the sentence for which the attribution was computed
    """
    assert attr.dim() == 1, "Attribution should be for just one sentence (without padding!)"
    assert (
        attr.shape[0] == input_emb.shape[1]
    ), "Attribution should contain an equal number of tokens as the input embedding!"
    replaced_embed = input_emb.clone()
    num_replace = round(
        (attr.shape[0] - 2) * k
    )  # minus 2, since cls and sep token should not count to the number of tokens for the ablation
    indices_replace = torch.topk(attr, num_replace).indices
    assert 0 not in indices_replace, "Should not replace CLS embedding"
    assert attr.shape[0] - 1 not in indices_replace, "Should not replace SEP embedding"
    replaced_embed[0, indices_replace] = replacement_emb

    return replaced_embed


def get_avg_scores(scores: Dict[str, Dict[float, List[float]]]) -> Dict[str, Dict[float, float]]:
    """
    Get average scores from a list of scores
    """
    avg_comps: Dict[str, Dict[float, float]] = {
        baseline_str: defaultdict(float) for baseline_str in scores.keys()
    }
    for baseline_str in scores.keys():
        for k in scores[baseline_str].keys():
            avg_comps[baseline_str][k] = np.mean(scores[baseline_str][k])
    
    return avg_comps
