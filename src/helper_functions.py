import torch
import pickle
from captum.attr._utils.approximation_methods import approximation_parameters
from functools import cache
from torch import Tensor
from typing import Tuple, List


def stringify_label(label: int) -> str:
    return "positive" if label == 1 else "negative"


def hash_tensor(tensor: Tensor) -> str:
    return str(hash(tuple(tensor.tolist())))


def construct_word_embedding(model, model_str: str, input_ids: Tensor):
    return getattr(model, model_str).embeddings.word_embeddings(input_ids)


def get_word_embeddings(model, model_str: str, trim_unused: bool = False):
    """
    word embeddings of whole vocabulary. Trim unused only works for the bert vocab.

    :param trim_unused: If false, embeddings of unused tokens are replaced so that they are out of the way (all 1s)
    If True, these embeddings are simply removed. Note that you cannot use those embeddings for decoding then.
    """
    # [0:1]     PAD
    # [1:100]   unused
    # [100:104] UNK CLS SEP MASK
    # [104:999] unused
    # [999:]    vocabulary
    with torch.no_grad():
        orig = getattr(model, model_str).embeddings.word_embeddings.weight.clone()
        if trim_unused:
            mod = torch.cat((orig[0:1], orig[100:104], orig[999:]))
        else:
            mod = orig
            mod[1:100] = 1.0
            mod[104:999] = 1.0
        return mod


def predict(model, inputs_embeds, attention_mask=None):
    return model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)[0]


def load_mappings(model_str, knn_nbrs=500):
    with open(f"knn/{model_str}_{knn_nbrs}.pkl", "rb") as f:
        [word_idx_map, word_features, adj] = pickle.load(f)
    word_idx_map = dict(word_idx_map)

    return word_idx_map, word_features, adj


def nn_forward_fn(
    model,
    model_str,
    input_embed,
    attention_mask=None,
    position_embed=None,
    type_embed=None,
    return_all_logits=True,
):
    # embeds = input_embed + position_embed
    embeds = input_embed
    embeds = getattr(model, model_str).embeddings.dropout(
        getattr(model, model_str).embeddings.LayerNorm(embeds)
    )
    pred = predict(model, embeds, attention_mask=attention_mask)
    if return_all_logits:
        return pred
    else:
        return pred.max(1).values


def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions
