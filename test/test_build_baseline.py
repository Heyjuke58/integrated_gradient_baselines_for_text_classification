import unittest
from parameterized import parameterized_class
from src.baseline_builder import BaselineBuilder
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.parse_arguments import MODEL_STRS, BASELINE_STRS
from src.helper_functions import construct_word_embedding

import torch


@parameterized_class([{"model_str": "distilbert"}, {"model_str": "bert"}])
class TestBaselineBuilder(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_STRS[self.model_str])
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_STRS[self.model_str], return_dict=False
        )
        self.bb = BaselineBuilder(self.model, self.model_str, self.tokenizer, seed=33)

    def test_build_baseline(self):
        """
        Given multiple input sentences, check for the constant CLS, SEP, PAD tokens. These should remain the same in the baseline
        for any given input and any baseline method. Check that all other tokens are changed
        (baseline is never equal to input).
        """
        # check for cls, sep, pad tokens at the right places when given multiple strings
        sentences = ["hello world!", "what a great movie."]
        # [[101, 7592, 2088,  999,  102],
        #  [101, 2054, 1037, 2307, 3185, 1012, 102]]
        # [['[CLS]', 'hello', 'world', '!', '[SEP]'],
        #  ['[CLS]', 'what', 'a', 'great', 'movie', '.', '[SEP]']]
        # c...spp
        # c.....s

        inputs_tok = self.tokenizer(sentences, padding=True, return_tensors="pt")
        inputs_emb = construct_word_embedding(self.model, self.model_str, inputs_tok["input_ids"])

        b_types = ["furthest_embed", "pad_embed", "blurred_embed"]
        # b_types = BASELINE_STRS
        for b_type in b_types:
            baselines_emb = self.bb.build_baseline(inputs_emb, b_type=b_type)
            self.assertEqual(inputs_emb.shape, baselines_emb.shape)

            self.assertTrue(
                torch.all(baselines_emb[:, 0] == self.bb.cls_emb.repeat(len(sentences), 1, 1))
            )
            self.assertTrue(torch.all(baselines_emb[0, 5:] == self.bb.pad_emb.repeat(2, 1, 1)))
            self.assertTrue(torch.all(baselines_emb[0, 4] == self.bb.sep_emb))
            self.assertTrue(torch.all(baselines_emb[1, -1] == self.bb.sep_emb))

            # check that baseline is otherwise different to original
            equal_mask = torch.tensor(
                [[1, 0, 0, 0, 1, 1, 1], [1, 0, 0, 0, 0, 0, 1]], dtype=torch.bool
            )
            self.assertTrue(torch.all(torch.all(baselines_emb == inputs_emb, dim=2) == equal_mask))

    def test_furthest_embed(self):
        """
        A fake embedding of all 0.1 should be given a baseline of all ~-0.25 (emb_min)
        """
        input_emb = torch.full((6, 768), 0.1)
        emb_min = self.bb.emb_mean - 2.58 * self.bb.emb_std
        baseline_emb = self.bb._furthest_embed(input_emb)

        self.assertLess(emb_min, 0.2)
        self.assertTrue(torch.all(baseline_emb == emb_min))

    def test_blurred_embed(self):
        """
        In a 3-dimensional embedding, the values in the different dimensions should be independent of eachother.
        """
        input_emb = torch.zeros((6, 3), dtype=float)
        input_emb[:, 0] = -1000.0
        input_emb[:, 2] = 1000.0
        baseline_emb = self.bb._blurred_embed(input_emb)

        self.assertTrue(torch.allclose(baseline_emb, input_emb))

        # input_emb = torch.rand((6, 768))

    def test_deterministic(self):
        """
        With the same seed, randomized baselines for the same sentence should be the same.
        """
        bb1 = BaselineBuilder(self.model, self.model_str, self.tokenizer, seed=33)
        bb2 = BaselineBuilder(self.model, self.model_str, self.tokenizer, seed=33)
        sentence = "hello world!"
        input_tok = self.tokenizer(sentence, padding=True, return_tensors="pt")
        input_emb = construct_word_embedding(self.model, self.model_str, input_tok["input_ids"])

        bl_types = ["uniform", "gaussian"]
        for bl_type in bl_types:
            bl1_emb = bb1.build_baseline(input_emb, b_type=bl_type)
            bl2_emb = bb2.build_baseline(input_emb, b_type=bl_type)
            self.assertTrue(torch.equal(bl1_emb, bl2_emb))

    def test_uniform(self):
        """
        Embeddings should be different to another. 
        """
        # TODO
        return
        input_emb = torch.zeros((6, 768), 0.1)
        bl_emb = self.bb._uniform(input_emb)


if __name__ == "__main__":
    unittest.main()
