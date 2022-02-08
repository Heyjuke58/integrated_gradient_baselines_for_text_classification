from functools import partial
from torch.nn.functional import softmax
from typing import List, Dict, Union
from collections import defaultdict

import torch
from datasets.load import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.baseline_builder import BaselineBuilder
from src.helper_functions import (
    construct_word_embedding,
    get_word_embeddings,
    nn_forward_fn,
    load_mappings,
    predict,
    stringify_label,
)
from src.parse_arguments import MODEL_STRS, parse_arguments
from src.visualization import embedding_histogram, visualize_attrs, visualize_ablation_scores
from src.dig import DiscretizedIntegratedGradients
from src.monotonic_paths import scale_inputs
from src.custom_ig import CustomIntegratedGradients
from src.ablation_evaluation import (
    calculate_suff,
    calculate_comp,
    get_avg_scores,
    calculate_log_odds,
)
from src.token_embedding_helper import TokenEmbeddingHelper

# K's for TopK ablation tests
K = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(
    examples: List[int],
    baselines: List[str],
    models: List[str],
    version_ig: str,
    steps: int,
    seed: int,
    viz_attr: bool,
    viz_topk: bool,
) -> None:
    """
    :param examples: list of indices for samples from the sst2 validation set to be classified and explained.
    :param baselines: list of
    :param models: keys for MODEL_STRS. What models should be used for classification and to be explained.
    """
    dataset = load_dataset("glue", "sst2", split="test")
    print(DEV)

    for model_str in models:
        print(f"MODEL: {model_str}")
        # from the DIG paper (for sst2):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_STRS[model_str])
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_STRS[model_str], return_dict=False
        ).to(DEV)
        cls_emb = construct_word_embedding(
            model, model_str, torch.tensor([[tokenizer.cls_token_id]]).to(DEV)
        )
        sep_emb = construct_word_embedding(
            model, model_str, torch.tensor([[tokenizer.sep_token_id]]).to(DEV)
        )
        pad_emb = construct_word_embedding(
            model, model_str, torch.tensor([[tokenizer.pad_token_id]]).to(DEV)
        )

        token_emb_helper = TokenEmbeddingHelper(model, model_str)

        # plot histogram
        # all_word_embeddings = get_word_embeddings(model, model_str)
        # embedding_histogram(all_word_embeddings)
        # continue

        # choose IG version
        ig: Union[CustomIntegratedGradients, DiscretizedIntegratedGradients]
        if version_ig == "ig":
            ig = CustomIntegratedGradients(partial(nn_forward_fn, model, model_str))
        elif version_ig == "dig":
            ig = DiscretizedIntegratedGradients(partial(nn_forward_fn, model, model_str))
            # get knn auxiliary data
            auxiliary_data = load_mappings(model_str)

        # Comprehensiveness:
        comps: Dict[str, Dict[float, List[float]]] = {
            baseline_str: defaultdict(list) for baseline_str in baselines
        }
        # Log Odds:
        log_odds: Dict[str, Dict[float, List[float]]] = {
            baseline_str: defaultdict(list) for baseline_str in baselines
        }

        print(f"USED EXAMPLES:")
        [
            print(
                f"[{stringify_label(dataset[example]['label'])}]: {dataset[example]['sentence']}"
                for example in examples
            )
        ]

        l = len(examples)
        for example in examples:
            print(f"EXAMPLE: {example + 1}/{l}")
            x = dataset[example]
            input = tokenizer(x["sentence"], padding=True, return_tensors="pt").to(DEV)
            input_ids = input["input_ids"][0].cpu()
            words = tokenizer.convert_ids_to_tokens(list(map(int, input_ids)))
            # formatted_input = (input["input_ids"], input["attention_mask"])
            input_emb = construct_word_embedding(model, model_str, input["input_ids"]).to(DEV)
            prediction = predict(model, input_emb, input["attention_mask"])
            prediction_probs = softmax(prediction, dim=1)
            prediction_str = (
                stringify_label(torch.argmax(prediction_probs).item())
                + f" ({(torch.max(prediction_probs).item() * 100):.2f}%)"
            )

            bl_attrs = {}
            for baseline_str in baselines:
                print(f"BASELINE: {baseline_str}")
                bb = BaselineBuilder(
                    model_str, seed, token_emb_helper, cls_emb, sep_emb, pad_emb, DEV
                )
                baseline = bb.build_baseline(input_emb, b_type=baseline_str).to(DEV)

                if version_ig == "ig":
                    # attributions for a batch of sentences
                    (attrs,), (word_paths,) = ig._attribute(
                        inputs=input_emb, baselines=baseline, n_steps=steps
                    )
                    word_paths_discretized = ()
                elif version_ig == "dig":
                    # attributions for one single sentence
                    attrs = []
                    baseline_ids = [
                        token_emb_helper.get_token_id(base_emb) for base_emb in baseline[0]
                    ]
                    # TODO hack for pad token baseline
                    # baseline_ids = (
                    #     [tokenizer.cls_token_id]
                    #     + [tokenizer.pad_token_id] * (baseline.shape[1] - 2)
                    #     + [tokenizer.sep_token_id]
                    # )
                    scaled_features, word_paths = scale_inputs(
                        input_ids,
                        baseline_ids,
                        DEV,
                        auxiliary_data,
                        steps=steps - 2,
                        strategy="greedy",
                    )
                    # TODO: maybe add additional forward args to attribute
                    attrs = ig.attribute(scaled_features=scaled_features, n_steps=steps)
                else:
                    raise Exception(
                        f"IG version should be one of ['ig', 'dig']. Instead it is {version_ig}"
                    )

                summed_attrs = torch.sum(torch.abs(attrs), dim=2).squeeze(0)
                bl_attrs[baseline_str] = summed_attrs.detach().cpu().numpy()

            if viz_topk:
                for baseline_str, attr in bl_attrs.items():
                    for k in K:
                        comps[baseline_str][k].append(
                            calculate_comp(
                                torch.tensor(attr, device=DEV),
                                k,
                                bb.pad_emb,
                                model,
                                input_emb,
                                input["attention_mask"],
                                prediction,
                            )
                        )
                        log_odds[baseline_str][k].append(
                            calculate_log_odds(
                                torch.tensor(attr, device=DEV),
                                k,
                                bb.pad_emb,
                                model,
                                input_emb,
                                input["attention_mask"],
                                prediction,
                            )
                        )
            if viz_attr:
                visualize_attrs(
                    bl_attrs,
                    prediction_str,
                    model_str,
                    version_ig,
                    x["sentence"],
                    words,
                )
        if viz_topk:
            avg_comps: Dict[str, Dict[float, float]] = get_avg_scores(comps)
            visualize_ablation_scores(
                avg_comps, model_str, "comprehensiveness", len(examples), save_str=None
            )
            avg_log_odds: Dict[str, Dict[float, float]] = get_avg_scores(log_odds)
            visualize_ablation_scores(
                avg_log_odds, model_str, "log odds", len(examples), save_str=None
            )


if __name__ == "__main__":
    args = parse_arguments()
    main(*args)
