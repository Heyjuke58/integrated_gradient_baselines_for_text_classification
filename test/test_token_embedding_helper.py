import itertools
import unittest
from functools import partial
from collections import defaultdict

import numpy as np
import torch
from parameterized import parameterized_class
from src.baseline_builder import BaselineBuilder
from src.custom_ig import CustomIntegratedGradients
from src.helper_functions import construct_word_embedding, nn_forward_fn
from src.parse_arguments import MODEL_STRS
from src.token_embedding_helper import TokenEmbeddingHelper
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@parameterized_class([{"model_str": "distilbert"}, {"model_str": "bert"}])
class TestTokenEmbeddingHelper(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_STRS[self.model_str])
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_STRS[self.model_str], return_dict=False
        ).to(DEV)
        self.cls_emb = construct_word_embedding(
            self.model, self.model_str, torch.tensor([[self.tokenizer.cls_token_id]], device=DEV)
        )
        self.sep_emb = construct_word_embedding(
            self.model, self.model_str, torch.tensor([[self.tokenizer.sep_token_id]], device=DEV)
        )
        self.pad_emb = construct_word_embedding(
            self.model, self.model_str, torch.tensor([[self.tokenizer.pad_token_id]], device=DEV)
        )
        self.token_emb_helper = TokenEmbeddingHelper(self.model, self.model_str)
        self.bb = BaselineBuilder(
            self.model,
            self.model_str,
            33,
            self.token_emb_helper,
            self.cls_emb,
            self.sep_emb,
            self.pad_emb,
            DEV,
        )

        sentence = "good movie ."
        self.inputs_tok = self.tokenizer(sentence, padding=True, return_tensors="pt").to(DEV)
        assert self.inputs_tok["input_ids"].shape == (1, 5)

        self.input_emb = construct_word_embedding(
            self.model, self.model_str, self.inputs_tok["input_ids"]
        )
        assert self.input_emb.shape == (1, 5, 768)
        self.baseline = self.bb.build_baseline(self.input_emb, b_type="pad_embed").to(DEV)

    def test_get_closest_by_token_emb(self):
        # test whether an actual embedding of a word is returned from the closest by function as one would expect, since the distance is 0
        orig_emb = self.input_emb[0, 0].clone()

        nearest_emb = self.token_emb_helper.get_closest_by_token_embed_for_embed(orig_emb)

        self.assertTrue(torch.equal(self.input_emb[0, 0], nearest_emb))

        # test whether the original embedding with minimal noise added is returned from the closest by function as one would expect, since the distance is very low
        with_minimal_noise = self.input_emb[0, 0].clone()
        with_minimal_noise[0] += 1e-6
        self.assertFalse(torch.equal(self.input_emb[0, 0], with_minimal_noise))

        nearest_emb = self.token_emb_helper.get_closest_by_token_embed_for_embed(with_minimal_noise)

        self.assertTrue(torch.equal(self.input_emb[0, 0], nearest_emb))

    def test_correct_dict_keys(self):
        # test emb to token id
        emb = self.input_emb[0][1].clone()
        corresponding_tok_id = self.inputs_tok["input_ids"][0, 1]

        self.assertTrue(corresponding_tok_id == self.token_emb_helper.get_token_id(emb))

        # test token id to emb
        emb = self.input_emb[0][2].clone()
        corresponding_tok_id = self.inputs_tok["input_ids"][0, 2].item()

        self.assertTrue(torch.equal(emb, self.token_emb_helper.get_emb(corresponding_tok_id)))

    def test_top_3_distances(self):
        emb = self.input_emb[0][1].clone()
        corresponding_tok_id = self.inputs_tok["input_ids"][0, 1]

        top3 = self.token_emb_helper.get_topk_closest_by_token_embed_for_embed(
            topk=3, embedding=emb, tokenizer=self.tokenizer
        )
        top10 = self.token_emb_helper.get_topk_closest_by_token_embed_for_embed(
            topk=10, embedding=emb, tokenizer=self.tokenizer
        )

        # check whether top3 has only smaller values than top10 after the first 3 elements
        for x, y in itertools.product(
            [dist for _, _, dist, _ in top3], [dist for _, _, dist, _ in top10[3:]]
        ):
            self.assertTrue(x < y)

    def test_top3_distances_from_good_to_pad(self):
        # tests whether the discretized word path for IG is correctly built, by reassuring the discretized token has the shortest distance to the interpolation embedding
        ig = CustomIntegratedGradients(partial(nn_forward_fn, self.model, self.model_str))
        (attrs,), (word_paths,) = ig.attribute(
            inputs=self.input_emb,
            baselines=self.baseline,
            n_steps=10,
            method="riemann_trapezoid",
            target=0,
        )
        wp_disc_emb = defaultdict(list)
        for word_path in word_paths:
            for i, word in enumerate(word_path):
                wp_disc_emb[i].append(
                    self.token_emb_helper.get_closest_by_token_embed_for_embed(word)
                    .detach()
                    .cpu()
                    .numpy()
                )
        wp_disc_actual_words = {
            i: self.tokenizer.convert_ids_to_tokens(
                [self.token_emb_helper.get_token_id(word) for word in word_path]
            )
            for i, word_path in wp_disc_emb.items()
        }
        wp_disc_token_ids = {
            i: [self.token_emb_helper.get_token_id(word) for word in word_path]
            for i, word_path in wp_disc_emb.items()
        }
        for j, word_emb in enumerate(word_paths[:, 1, :]):
            top3 = self.token_emb_helper.get_topk_closest_by_token_embed_for_embed(
                3, word_emb, self.tokenizer
            )
            self.assertTrue(top3[0][0] == wp_disc_token_ids[1][j])
