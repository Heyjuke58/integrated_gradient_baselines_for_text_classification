import unittest
import torch
from parameterized import parameterized_class
from src.helper_functions import construct_word_embedding
from src.baseline_builder import BaselineBuilder
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.parse_arguments import MODEL_STRS
from src.ablation_evaluation import replace_k_percent


@parameterized_class([{"model_str": "distilbert"}, {"model_str": "bert"}])
class TestAblationEvaluation(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_STRS[self.model_str])
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_STRS[self.model_str], return_dict=False
        )
        self.pad_emb = construct_word_embedding(
            self.model, self.model_str, torch.tensor([[self.tokenizer.pad_token_id]])
        )

        # sentence = "This has been the worst movie in the history of movies, maybe ever."
        sentence = "One two three four five six seven eight nine ten"
        inputs_tok = self.tokenizer(sentence, padding=True, return_tensors="pt")
        assert inputs_tok["input_ids"].shape == (1, 12)

        # this has not really gone through the model but this does not matter for the tests,
        # we just need an embedding:
        self.input_emb = construct_word_embedding(
            self.model, self.model_str, inputs_tok["input_ids"]
        )
        assert self.input_emb.shape == (1, 12, 768)
        self.replacement_emb = self.pad_emb

    def test_replace_k_percent(self):
        # the last words are the most important:
        attr = torch.zeros(12)
        attr[1:-1] = torch.arange(1, 11) / 10
        masked_emb = replace_k_percent(attr, 0, self.replacement_emb, self.input_emb)
        self.assertTrue(torch.equal(self.input_emb, masked_emb))

        # 9% should round up to 1 of 10 words (similar with 14%)
        masked_emb = replace_k_percent(attr, 0.09, self.replacement_emb, self.input_emb)
        masked_emb2 = replace_k_percent(attr, 0.14, self.replacement_emb, self.input_emb)
        expected_emb = torch.clone(self.input_emb)
        expected_emb[0, -2] = self.replacement_emb
        self.assertTrue(torch.equal(masked_emb, expected_emb))
        self.assertTrue(torch.equal(masked_emb2, expected_emb))
        
        # 50% of tokens replaced
        masked_emb = replace_k_percent(attr, 0.5, self.replacement_emb, self.input_emb)
        expected_emb = torch.clone(self.input_emb)
        expected_emb[0, 6:11] = self.replacement_emb
        self.assertTrue(torch.equal(masked_emb, expected_emb))