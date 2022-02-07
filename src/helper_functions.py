import torch
import pickle
from captum.attr._utils.approximation_methods import approximation_parameters
from functools import cache


def stringify_label(label: int) -> str:
    return "positive" if label == 1 else "negative"


def construct_word_embedding(model, model_str, input_ids):
    return getattr(model, model_str).embeddings.word_embeddings(input_ids)


def get_word_embeddings(model, model_str):
    return getattr(model, model_str).embeddings.word_embeddings.weight


@cache
def get_token_id_from_embedding(model, model_str, embedding):
    for i, word_embed in enumerate(get_word_embeddings(model, model_str)):
        if torch.equal(word_embed, embedding):
            return i
    raise KeyError("Embedding does not correspond to a token known in the vocabulary.")


def get_closest_by_token_id_for_embedding(model, model_str, embedding):
    min_dist = float("inf")
    token_id = None
    for i, word_embed in enumerate(get_word_embeddings(model, model_str)):
        dist = torch.cdist(word_embed, embedding) ** 2
        if dist < min_dist:
            min_dist = dist
            token_id = i
    return token_id


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
    return_all_logits=False,
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
