import unittest
from parameterized import parameterized_class
from src.build_baseline import BaselineBuilder
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.parse_arguments import MODEL_STRS


@parameterized_class(
    [
        {"model_str": "distilbert"},
        {"model_str": "bert"},
    ]
)
class TestBaselineBuilder(unittest.TestCase):
    def setUp(self) -> None:
        # model_str = "distilbert"  # TODO test both models always
        print(self.model_str)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_STRS[self.model_str])
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_STRS[self.model_str], return_dict=False
        )
        self.bb = BaselineBuilder(model, self.model_str, tokenizer)

    def test_build_baseline(self):
        pass

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
