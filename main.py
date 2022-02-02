from functools import partial
from typing import List

import matplotlib.pyplot as plt
import torch
from captum.attr import IntegratedGradients
from datasets.load import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.build_baseline import BaselineBuilder
from src.helper_functions import (construct_word_embedding,
                                  get_word_embeddings, nn_forward_fn)
from src.parse_arguments import MODEL_STRS, parse_arguments
from src.visualization import embedding_histogram, visualize_attrs


def main(examples: List[int], baselines: List[str], models: List[str]) -> None:
    """
    :param examples: list of indices for samples from the sst2 validation set to be classified and explained.
    :param baselines: list of
    :param models: keys for MODEL_STRS. What models should be used for classification and to be explained.
    """
    dataset = load_dataset("glue", "sst2", split="validation")

    for model_str in models:
        print(f"MODEL: {model_str}")
        # from the DIG paper (for sst2):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_STRS[model_str])
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_STRS[model_str], return_dict=False
        )

        # plot histogram
        all_word_embeddings = get_word_embeddings(model, model_str)
        # embedding_histogram(all_word_embeddings)
        # continue

        pad_token_id = tokenizer.pad_token_id
        sep_token_id = tokenizer.sep_token_id
        cls_token_id = tokenizer.cls_token_id

        ig = IntegratedGradients(partial(nn_forward_fn, model, model_str))
        x = dataset[0]
        # x = dataset[examples]
        input = tokenizer(x["sentence"], padding=True, return_tensors="pt")
        words = tokenizer.convert_ids_to_tokens(list(map(int, input["input_ids"][0])))
        formatted_input = (input["input_ids"], input["attention_mask"])
        input_emb = construct_word_embedding(model, model_str, input["input_ids"])

        # create the baseline:
        bb = BaselineBuilder(model, model_str, tokenizer)
        for baseline_str in baselines:
            print(f"BASELINE: {baseline_str}")
            # baseline = bb.build_baseline(input_emb, b_type=baseline_str)
            baseline = construct_word_embedding(
                model,
                model_str,
                torch.tensor([[cls_token_id] + [pad_token_id] * (len(words) - 2) + [sep_token_id]]),
            )
            # attributions for one whole sentence:
            attrs = ig.attribute(inputs=input_emb, baselines=baseline)

            summed_attrs = torch.sum(torch.abs(attrs), dim=2).squeeze(0)
            # summed_attrs = torch.sum(attrs, dim=2).squeeze(0)
            print(summed_attrs)

            # visualize_attrs(summed_attrs.detach().numpy(), words, save_str=model_str + "_absolute")
            visualize_attrs(summed_attrs.detach().numpy(), words)


if __name__ == "__main__":
    args = parse_arguments()
    main(*args)
