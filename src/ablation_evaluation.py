import torch
from torch import Tensor
from src.helper_functions import predict


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
    # TODO softmax on probs
    replaced_embed = input_emb.clone()
    num_replace = round(attr.shape[0] * k)
    indicies_replace = torch.topk(attr, num_replace).indices
    replaced_embed[0, indicies_replace] = replacement_emb

    new_pred = predict(model, replaced_embed, attention_mask).squeeze(0)
    pred_i = torch.argmax(prediction).item()
    return -(new_pred[pred_i] - torch.max(prediction)).item()
