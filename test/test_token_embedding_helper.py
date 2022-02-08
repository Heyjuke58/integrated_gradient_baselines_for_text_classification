import unittest
import torch
from parameterized import parameterized_class
from src.helper_functions import construct_word_embedding
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.parse_arguments import MODEL_STRS
from src.token_embedding_helper import TokenEmbeddingHelper


@parameterized_class([{"model_str": "distilbert"}, {"model_str": "bert"}])
class TestTokenEmbeddingHelper(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_STRS[self.model_str])
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_STRS[self.model_str], return_dict=False
        )
        self.token_emb_helper = TokenEmbeddingHelper(self.model, self.model_str)

        # sentence = "This has been the worst movie in the history of movies, maybe ever."
        sentence = "One two three four five six seven eight nine ten"
        self.inputs_tok = self.tokenizer(sentence, padding=True, return_tensors="pt")
        assert self.inputs_tok["input_ids"].shape == (1, 12)

        # this has not really gone through the model but this does not matter for the tests,
        # we just need an embedding:
        self.input_emb = construct_word_embedding(
            self.model, self.model_str, self.inputs_tok["input_ids"]
        )
        assert self.input_emb.shape == (1, 12, 768)

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
