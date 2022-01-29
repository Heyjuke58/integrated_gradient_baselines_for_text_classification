from typing import List

import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.parse_arguments import parse_arguments, MODEL_STRS, BASELINE_STRS


dataset = load_dataset("glue", "sst2", split="test")


# from the DIG paper (for sst2):
# bert
tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")
model = AutoModelForSequenceClassification.from_pretrained(
    "textattack/bert-base-uncased-SST-2", return_dict=False
)
# distilbert
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english", return_dict=False
)


x = dataset[0]
input = tokenizer(x["sentence"], padding=True, return_tensors="pt")

decoded = tokenizer.decode(input["input_ids"])
print(input)
out = model(**input)

print(out)
out_label = torch.max(out.logits)


def main(examples: List[int], baselines: List[str], models: List[str]) -> None:
    for model_str in models:
        pass


if __name__ == "__main__":
    args = parse_arguments()
    main(*args)
