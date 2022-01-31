from typing import List

from captum.attr import IntegratedGradients
from functools import partial

import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.parse_arguments import parse_arguments
from src.helper_functions import construct_word_embedding, nn_forward_fn
from src.visualization import visualize_attrs


def main(examples: List[int], baselines: List[str], models: List[str]) -> None:
    dataset = load_dataset("glue", "sst2", split="validation")

    for model_str in models:
        print(f"MODEL: {model_str}")
        # from the DIG paper (for sst2):
        tokenizer = AutoTokenizer.from_pretrained(model_str)
        model = AutoModelForSequenceClassification.from_pretrained(model_str, return_dict=False)
        ig = IntegratedGradients(partial(nn_forward_fn, model))
        x = dataset[0]
        input = tokenizer(x["sentence"], padding=True, return_tensors="pt")
        words = tokenizer.convert_ids_to_tokens(list(map(int, input["input_ids"][0])))
        formatted_input = (input["input_ids"], input["attention_mask"])
        input_emb = construct_word_embedding(model, model_str, input["input_ids"])
        # attributions for one whole sentence:
        attrs = ig.attribute(inputs=input_emb)

        summed_attrs = torch.sum(attrs, dim=2).squeeze(0)
        print(summed_attrs)
        
        visualize_attrs(summed_attrs.detach().numpy(), words)

        for baseline_str in baselines:
            for example in examples:
                pass


if __name__ == "__main__":
    args = parse_arguments()
    main(*args)
