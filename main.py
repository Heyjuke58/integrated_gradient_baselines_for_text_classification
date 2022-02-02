from functools import partial
from typing import List

import torch
from captum.attr import IntegratedGradients
from datasets.load import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.build_baseline import BaselineBuilder
from src.helper_functions import (
    construct_word_embedding,
    get_word_embeddings,
    nn_forward_fn,
    load_mappings,
)
from src.parse_arguments import MODEL_STRS, parse_arguments
from src.visualization import embedding_histogram, visualize_attrs
from src.dig import DiscretetizedIntegratedGradients
from src.monotonic_paths import scale_inputs


def main(
    examples: List[int], baselines: List[str], models: List[str], version_ig: str, steps: int
) -> None:
    """
    :param examples: list of indices for samples from the sst2 validation set to be classified and explained.
    :param baselines: list of
    :param models: keys for MODEL_STRS. What models should be used for classification and to be explained.
    """
    dataset = load_dataset("glue", "sst2", split="validation")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    for model_str in models:
        print(f"MODEL: {model_str}")
        # from the DIG paper (for sst2):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_STRS[model_str])
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_STRS[model_str], return_dict=False
        )

        # plot histogram
        # all_word_embeddings = get_word_embeddings(model, model_str)
        # embedding_histogram(all_word_embeddings)
        # continue

        # choose IG version
        if version_ig == "ig":
            ig: IntegratedGradients = IntegratedGradients(partial(nn_forward_fn, model, model_str))
        elif version_ig == "dig":
            # TODO: add DIG
            ig: DiscretetizedIntegratedGradients = DiscretetizedIntegratedGradients(
                partial(nn_forward_fn, model, model_str)
            )
            # get knn auxiliary data
            auxiliary_data = load_mappings(model_str)

        x = dataset[examples]
        input = tokenizer(x["sentence"], padding=True, return_tensors="pt")
        words = tokenizer.convert_ids_to_tokens(list(map(int, input["input_ids"][0])))
        formatted_input = (input["input_ids"], input["attention_mask"])
        input_emb = construct_word_embedding(model, model_str, input["input_ids"])

        bb = BaselineBuilder(model, model_str, tokenizer)
        bl_attrs = {}
        for baseline_str in baselines:
            print(f"BASELINE: {baseline_str}")
            baseline = bb.build_baseline(input_emb, b_type=baseline_str)

            if version_ig == "ig":
                # attributions for a batch of sentences
                attrs = ig.attribute(inputs=input_emb, baselines=baseline)
            elif version_ig == "dig":
                # attributions for one single sentence
                attrs = []
                for input_example, baseline_example in zip(input_emb, baseline):
                    # TODO: either: adapt scale_inputs to correctly build path from baseline embedding
                    # TODO: or: get baseline token ids from baseline embedding
                    scaled_features = scale_inputs(
                        input_example,
                        baseline_example,
                        device,
                        auxiliary_data,
                        steps=steps,
                        strategy="greedy",
                    )
                    # TODO: maybe add additional forward args to attribute
                    attr = ig.attribute(scaled_features=scaled_features, n_steps=steps)
                    attrs.append(attr)
                attrs = torch.cat(attrs, dim=0)
            summed_attrs = torch.sum(torch.abs(attrs), dim=2).squeeze(0)
            bl_attrs[baseline_str] = summed_attrs.detach().numpy()

        # visualize_attrs(summed_attrs.detach().numpy(), words, save_str=model_str + "_absolute")
        visualize_attrs(bl_attrs, words)


if __name__ == "__main__":
    args = parse_arguments()
    main(*args)
