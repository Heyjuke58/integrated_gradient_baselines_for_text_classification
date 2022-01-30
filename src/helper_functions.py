import torch

def construct_word_embedding(model, model_str, input_ids):
    return model.distilbert.embeddings.word_embeddings(input_ids)


def predict(model, inputs_embeds, attention_mask=None):
    return model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)[0]


def nn_forward_fn(
    model,
    input_embed,
    attention_mask=None,
    position_embed=None,
    type_embed=None,
    return_all_logits=False,
):
    # embeds = input_embed + position_embed
    embeds = input_embed
    embeds = model.distilbert.embeddings.dropout(model.distilbert.embeddings.LayerNorm(embeds))
    pred = predict(model, embeds, attention_mask=attention_mask)
    if return_all_logits:
        return pred
    else:
        return pred.max(1).values

def summarize_attributions(attributions):
	attributions = attributions.sum(dim=-1).squeeze(0)
	attributions = attributions / torch.norm(attributions)
	return attributions