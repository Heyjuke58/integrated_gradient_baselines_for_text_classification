from typing import Optional, List
from torchvision.transforms.functional import gaussian_blur

import torch
from torch import Tensor

from src.helper_functions import construct_word_embedding
from src.parse_arguments import BASELINE_STRS

# mean and std of all word embeddings for our two used models
# we need these for certain baseline computations
EMB_STATS = {
    "distilbert": {"mean": -0.03833248, "std": 0.046996452},
    "bert": {"mean": -0.028025009, "std": 0.042667598},
}


class BaselineBuilder:
    def __init__(
        self,
        model_str: str,
        seed: int,
        cls_emb: Tensor,
        sep_emb: Tensor,
        pad_emb: Tensor,
        device,
    ) -> None:
        """
        Create a specific type of baseline for IG for a sentence.
        Note that the baseline will always keep the [CLS] and [SEP] tokens, and these will also always stay constant
        in the "paths" for IG.

        :param model_str: One of ['distilbert', 'bert']
        :param seed: Seed for the random baseline generation
        :param cls_emb: Embedding for the cls token
        :param sep_emb: Embedding for the sep token
        :param pad_emb: Embedding for the pad token
        """
        self.cls_emb = cls_emb
        self.sep_emb = sep_emb
        self.pad_emb = pad_emb
        self.emb_mean = EMB_STATS[model_str]["mean"]
        self.emb_std = EMB_STATS[model_str]["std"]
        self.emb_max = self.emb_mean + 2.58 * self.emb_std
        self.emb_min = self.emb_mean - 2.58 * self.emb_std
        self.rngesus = torch.Generator(device=device).manual_seed(seed)
        self.device = device

    def build_baseline(self, input_emb: Tensor, b_type: str) -> Tensor:
        """
        Constructs baseline embedding tensor from input embedding tensor shape: (sentences, tokens, embedding) for a given baseline method (b_type)

        Args:
            input_emb (Tensor): Input embedding tensor
            b_type (str): baseline mehtod (must be one in BASELINE_STRS)

        Returns:
            Tensor: The baseline tensor
        """
        # build mask for the actual sentence tokens to be replaced by a baseline
        input_mask = torch.zeros(input_emb.shape[0:2], dtype=bool).to(self.device)
        baseline_emb = input_emb.detach().clone()
        for i, sentence in enumerate(baseline_emb):
            assert torch.equal(
                baseline_emb[i, 0], self.cls_emb[0, 0]
            ), "First embedding of input does not match cls token embedding."
            for j, word_emb in enumerate(sentence):
                if torch.equal(word_emb, self.sep_emb[0, 0]):
                    assert (
                        torch.equal(baseline_emb[i, j], self.sep_emb[0, 0]) and j != 0
                    ), "Input embedding does not contain sep token embedding."
                    input_mask[i, 1:j] = 1

        # get all sentences stripped of cls, sep and pad tokens for baseline creation
        input_sentences: List[Tensor] = []
        for i, sentence in enumerate(baseline_emb):
            input_sentences.append(sentence[input_mask[i]])

        assert (
            b_type in BASELINE_STRS
        ), f"{b_type} not a legal baseline method! Must be one of {BASELINE_STRS}."

        base_sentence_embs: Tensor = torch.Tensor([])
        if b_type == "furthest_embed":
            base_sentence_embs = [self._furthest_embed(s) for s in input_sentences]
        elif b_type == "blurred_embed":
            base_sentence_embs = [self._blurred_embed(s) for s in input_sentences]
        elif b_type == "flipped_blurred_embed":
            base_sentence_embs = [self._flipped_blurred_embed(s) for s in input_sentences]
        elif b_type == "both_blurred_embed":
            base_sentence_embs = [self._both_blurred_embed(s) for s in input_sentences]
        elif b_type == "pad_embed":
            base_sentence_embs = [self._pad_embed(s) for s in input_sentences]
        elif b_type == "uniform":
            base_sentence_embs = [self._uniform(s) for s in input_sentences]
        elif b_type == "gaussian":
            base_sentence_embs = [self._gaussian(s) for s in input_sentences]
        elif b_type == "furthest_word":
            base_sentence_embs = [self._furthest_word(s) for s in input_sentences]
        elif b_type == "avg_word_embed":
            base_sentence_embs = [self._avg_word_embed(s) for s in input_sentences]
        elif b_type == "avg_word":
            base_sentence_embs = [self._avg_word(s) for s in input_sentences]

        for i, base_sentence_emb in enumerate(base_sentence_embs):
            baseline_emb[i][input_mask[i]] = base_sentence_emb

        return baseline_emb

    def _furthest_embed(self, input_emb: Tensor) -> Tensor:
        """
        Given a sentence embedding, return the furthest point in the embedding space from it.
        Bounds of the embedding space are found by using the sigma-surrounding to exclude extreme values.

        Note: For all baseline methods like this one, the input_emb is required to be in the below specified format!

        :param input_embed: embeddings of words (not the surrounding cls, sep, or pad embeddings!)
        """
        # min and max vals are the bounds of the sigma-surrounding which includes ~99% of all values
        small_vals_mask = input_emb < self.emb_mean
        big_vals_mask = ~small_vals_mask

        baseline_emb = torch.zeros_like(input_emb, device=self.device)
        baseline_emb[small_vals_mask] = self.emb_max
        baseline_emb[big_vals_mask] = self.emb_min
        return baseline_emb

    def _blurred_embed(self, input_emb: Tensor) -> Tensor:
        """
        Blurs over the words in the input. Different embedding dimensions are blurred independently of another.
        """
        return gaussian_blur(input_emb.unsqueeze(0), kernel_size=[1, 3]).squeeze(0)

    def _flipped_blurred_embed(self, input_emb: Tensor) -> Tensor:
        """
        Words are independent, embeddings of each word are blurred.
        """
        return gaussian_blur(input_emb.unsqueeze(0), kernel_size=[21, 1]).squeeze(0)

    def _both_blurred_embed(self, input_emb: Tensor) -> Tensor:
        """
        Everything is blurred.
        """
        return gaussian_blur(input_emb.unsqueeze(0), kernel_size=[21, 3]).squeeze(0)

    def _pad_embed(self, input_emb: Tensor) -> Tensor:
        """
        Ignores the input and returns a baseline consisting of just the embeddings of the PAD token (as done in the DIG paper).
        """
        baseline_emb = torch.zeros_like(input_emb, device=self.device)
        for i, _ in enumerate(baseline_emb):
            baseline_emb[i] = self.pad_emb

        return baseline_emb

    def _uniform(self, input_emb: Tensor) -> Tensor:
        return torch.cuda.FloatTensor(input_emb.shape).uniform_(
            self.emb_min, self.emb_max, generator=self.rngesus
        )

    def _gaussian(self, input_emb: Tensor) -> Tensor:
        # return torch.FloatTensor(input_emb.shape).uniform_(self.emb_min, self.emb_max)
        return torch.clamp(
            torch.normal(mean=input_emb, std=1, generator=self.rngesus),
            min=self.emb_min,
            max=self.emb_max,
        )

    def _furthest_word(self, input_emb: Tensor) -> Tensor:
        raise NotImplementedError

    def _avg_word_embed(self, input_emb: Tensor) -> Tensor:
        raise NotImplementedError

    def _avg_word(self, input_emb: Tensor) -> Tensor:
        raise NotImplementedError
