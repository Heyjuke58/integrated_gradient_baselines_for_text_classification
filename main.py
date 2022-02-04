from functools import partial
from typing import List, Dict
from collections import defaultdict

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
    get_token_id_from_embedding,
    predict,
    stringify_label,
)
from src.parse_arguments import MODEL_STRS, parse_arguments
from src.visualization import embedding_histogram, visualize_attrs
from src.dig import DiscretetizedIntegratedGradients
from src.monotonic_paths import scale_inputs
from src.custom_ig import CustomIntegratedGradients
from src.ablation_evaluation import calculate_suff, calculate_comp


def main(
    examples: List[int],
    baselines: List[str],
    models: List[str],
    version_ig: str,
    steps: int,
    seed: int,
    viz_attr: bool,
    viz_comp: bool,
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
            ig: CustomIntegratedGradients = CustomIntegratedGradients(
                partial(nn_forward_fn, model, model_str)
            )
        elif version_ig == "dig":
            # TODO: add DIG
            ig: DiscretetizedIntegratedGradients = DiscretetizedIntegratedGradients(
                partial(nn_forward_fn, model, model_str)
            )
            # get knn auxiliary data
            auxiliary_data = load_mappings(model_str)

        # Comprehensiveness:
        comps: Dict[str, Dict[float, List[float]]] = {
            baseline_str: defaultdict(list) for baseline_str in baselines
        }

        print(f"USED EXAMPLES:")
        print(
            [
                f"[{stringify_label(dataset[example]['label'])}]: {dataset[example]['sentence']}"
                for example in examples
            ]
        )

        for example in examples:
            x = dataset[example]
            input = tokenizer(x["sentence"], padding=True, return_tensors="pt")
            input_ids = input["input_ids"][0]
            words = tokenizer.convert_ids_to_tokens(list(map(int, input_ids)))
            # formatted_input = (input["input_ids"], input["attention_mask"])
            input_emb = construct_word_embedding(model, model_str, input["input_ids"])
            true_label = stringify_label(x["label"])
            prediction = predict(model, input_emb, input["attention_mask"])
            prediction_str = stringify_label(torch.argmax(prediction).item())

            bl_attrs = {}
            for baseline_str in baselines:
                print(f"BASELINE: {baseline_str}")
                bb = BaselineBuilder(model, model_str, tokenizer, seed)
                baseline = bb.build_baseline(input_emb, b_type=baseline_str)

                if version_ig == "ig":
                    # attributions for a batch of sentences
                    (attrs,), (word_paths,) = ig._attribute(
                        inputs=input_emb, baselines=baseline, n_steps=steps
                    )
                elif version_ig == "dig":
                    # attributions for one single sentence
                    attrs = []
                    baseline_ids = [
                        get_token_id_from_embedding(model, model_str, base_emb)
                        for base_emb in baseline[0]
                    ]
                    scaled_features, word_paths = scale_inputs(
                        input_ids,
                        baseline_ids,
                        device,
                        auxiliary_data,
                        steps=steps - 2,
                        strategy="greedy",
                    )
                    # TODO: maybe add additional forward args to attribute
                    attrs = ig.attribute(scaled_features=scaled_features, n_steps=steps)

                summed_attrs = torch.sum(torch.abs(attrs), dim=2).squeeze(0)
                bl_attrs[baseline_str] = summed_attrs.detach().numpy()

            if viz_comp:
                for baseline_str, attr in bl_attrs.items():
                    for k in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
                        comps[baseline_str][k].append(
                            calculate_comp(
                                torch.tensor(attr),
                                k,
                                bb.pad_emb,
                                model,
                                input_emb,
                                input["attention_mask"],
                                prediction,
                            )
                        )

            # visualize_attrs(summed_attrs.detach().numpy(), words, save_str=model_str + "_absolute")
            if viz_attr:
                visualize_attrs(
                    bl_attrs,
                    prediction_str,
                    true_label,
                    model_str,
                    version_ig,
                    x["sentence"],
                    words,
                )
        if viz_comp:
            # visualize_comps(comps)
            pass


if __name__ == "__main__":
    args = parse_arguments()
    main(*args)
