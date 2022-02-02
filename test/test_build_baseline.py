import unittest
from parameterized import parameterized_class
from src.build_baseline import BaselineBuilder
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.parse_arguments import MODEL_STRS, BASELINE_STRS
from src.helper_functions import construct_word_embedding

import torch


@parameterized_class([{"model_str": "distilbert"}, {"model_str": "bert"}])
class TestBaselineBuilder(unittest.TestCase):
    def setUp(self) -> None:
        # model_str = "distilbert"  # TODO test both models always
        print(self.model_str)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_STRS[self.model_str])
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_STRS[self.model_str], return_dict=False
        )
        self.bb = BaselineBuilder(self.model, self.model_str, self.tokenizer)

    def test_build_baseline(self):
        """
        Check for the constant CLS, SEP, PAD tokens. These should remain the same in the baseline
        for any given input and any baseline method. Check that all other tokens are changed
        (baseline is never equal to input).
        """
        # check for cls, sep, pad tokens at the right places
        sentences = ["hello world!", "what a great movie."]
        # [[101, 7592, 2088,  999,  102],
        #  [101, 2054, 1037, 2307, 3185, 1012, 102]]
        # [['[CLS]', 'hello', 'world', '!', '[SEP]'],
        #  ['[CLS]', 'what', 'a', 'great', 'movie', '.', '[SEP]']]
        # c...spp
        # c.....s

        inputs_tok = self.tokenizer(sentences, padding=True, return_tensors="pt")
        inputs_emb = construct_word_embedding(self.model, self.model_str, inputs_tok["input_ids"])

        b_types = ["furthest_embedding", "pad_token", "blurred_embedding"]
        # b_types = BASELINE_STRS
        for b_type in b_types:
            baselines_emb = self.bb.build_baseline(inputs_emb, b_type=b_type)
            self.assertEqual(inputs_emb.shape, baselines_emb.shape)

            self.assertTrue(
                torch.all(baselines_emb[:, 0] == self.bb.cls_emb.repeat(len(sentences), 1, 1))
            )
            self.assertTrue(
                torch.all(baselines_emb[0, 5:] == self.bb.pad_emb.repeat(2, 1, 1))
            )
            self.assertTrue(torch.all(baselines_emb[0, 4] == self.bb.sep_emb))
            self.assertTrue(torch.all(baselines_emb[1, -1] == self.bb.sep_emb))
            equal_mask = torch.tensor(
                [[1, 0, 0, 0, 1, 1, 1], [1, 0, 0, 0, 0, 0, 1]], dtype=torch.bool
            )
            self.assertTrue(torch.all(torch.all(baselines_emb == inputs_emb, dim=2) == equal_mask))

    # def test_upper(self):
    #     self.assertEqual("foo".upper(), "FOO")
    #
    # def test_isupper(self):
    #     self.assertTrue("FOO".isupper())
    #     self.assertFalse("Foo".isupper())
    #
    # def test_split(self):
    #     s = "hello world"
    #     self.assertEqual(s.split(), ["hello", "world"])
    #     # check that s.split fails when the separator is not a string
    #     with self.assertRaises(TypeError):
    #         s.split(2)


if __name__ == "__main__":
    unittest.main()
