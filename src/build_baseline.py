from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch import Tensor
import torch
from src.helper_functions import construct_word_embedding
from typing import Optional
from src.parse_arguments import BASELINE_STRS

EMB_STATS = {
    "distilbert": {
        "mean": -0.03833248,
        "std": 0.046996452,
    },
    "bert": {
        "mean": -0.028025009,
        "std": 0.042667598,
    },
}


class BaselineBuilder:
    def __init__(
        self,
        model: AutoModelForSequenceClassification,
        model_str: str,
        tokenizer: AutoTokenizer,
    ) -> None:
        """
        Create a specific type of baseline for IG for a sentence.
        Note that the baseline will always keep the [CLS] and [SEP] tokens, and these will also always stay constant
        in the "paths" for IG.

        :param b_type: Key for ... . Type of baseline that should be generated.
        """
        self.cls_emb = construct_word_embedding(
            model, model_str, torch.tensor([[tokenizer.cls_token_id]])
        )
        self.sep_emb = construct_word_embedding(
            model, model_str, torch.tensor([[tokenizer.sep_token_id]])
        )
        self.pad_emb = construct_word_embedding(
            model, model_str, torch.tensor([[tokenizer.pad_token_id]])
        )
        self.emb_mean = EMB_STATS[model_str]["mean"]
        self.emb_std = EMB_STATS[model_str]["std"]

    def build_baseline(self, input_emb: Tensor, b_type: str):
        assert (
            input_emb.shape[0] == 1
        ), f"Input embedding should have a shape of (1, tokens, embedding size). Instead it has {input_emb.shape}"
        assert (
            input_emb[0, 0] == self.cls_emb
        ), "First embedding of input does not match cls token embedding."
        sep_idx = 0
        for i, row in enumerate(input_emb[0]):
            if torch.eq(row, self.sep_emb):
                sep_idx = i
        assert (
            input_emb[0, sep_idx] == self.sep_emb and sep_idx != 0
        ), "Input embedding does not contain sep token embedding."

        # get number of appended padding tokens
        num_appended_pad_tokens = input_emb.shape[1] - (sep_idx + 1)
        pad_emb = torch.cat([self.pad_emb] * num_appended_pad_tokens, dim=1)

        input_emb = input_emb[1:sep_idx]  # get only relevant tokens for the baseline creation
        num_tokens = input_emb.shape[1]  # without cls, sep and appended pad tokens

        assert (
            b_type in BASELINE_STRS
        ), f"{b_type} not a legal baseline method! Must be one of {BASELINE_STRS}."

        base_emb: Tensor = torch.Tensor([])
        if b_type == "furthest_embedding":
            base_emb = self.furthest_embedding(input_emb)
        elif b_type == "blurred_embedding":
            pass
        elif b_type == "pad_token":
            pass
        elif b_type == "uniform":
            pass
        elif b_type == "gaussian":
            pass
        elif b_type == "furthest_word":
            pass
        elif b_type == "average_word_embedding":
            pass
        elif b_type == "average_word":
            pass

        baseline = torch.cat((self.cls_emb, base_emb, self.sep_emb, pad_emb), dim=0)

        return baseline

    def furthest_embedding(self, input_emb: Tensor) -> Tensor:
        """
        Given a sentence embedding, return the furthest point in the embedding space from it.
        Bounds of the embedding space are found by using the sigma-surrounding to exclude extreme values.

        :param input_embed: embeddings of words (not the surrounding cls, sep, or pad embeddings!)
        """
        # min and max vals are the bounds of the sigma-surrounding which includes ~99% of all values
        emb_max = self.emb_mean + 2.58 * self.emb_std
        emb_min = self.emb_mean - 2.58 * self.emb_std
        small_vals_mask = input_emb < self.emb_mean
        big_vals_mask = ~small_vals_mask

        baseline_emb = torch.zeros_like(input_emb)
        baseline_emb[small_vals_mask] = emb_max
        baseline_emb[big_vals_mask] = emb_min
        return baseline_emb
