import argparse
from typing import List, Dict, Any, Tuple

MODEL_STRS = {
    # https://huggingface.co/textattack/bert-base-uncased-SST-2
    "bert": "textattack/bert-base-uncased-SST-2",
    # https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english
    "distilbert": "distilbert-base-uncased-finetuned-sst-2-english",
    # "sshleifer": "sshleifer/tiny-distilbert-base-uncased-finetuned-sst-2-english",
}
BASELINE_STRS = {
    "pad_token",
    "furthest_embedding",
    "blurred_embedding",
    "uniform",
    "gaussian",
    "furthest_word",
    "average_word_embedding",
    "average_word",
}


def parse_arguments() -> Tuple[List[int], List[str], List[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--examples",
        type=str,
        dest="examples",
        default="0,1,2,3",
        help="Example indices used for IG evaluation.",
    )
    parser.add_argument(
        "-b",
        "--baselines",
        type=str,
        dest="baselines",
        default="pad_token",
        help="Type of baseline to be used for Integrated Gradients.",
    )
    parser.add_argument(
        "-m",
        "--models",
        type=str,
        dest="models",
        default="distilbert,bert",
        help="Models that should be interpreted.",
    )

    args = parser.parse_args()
    args.examples = list(map(int, args.examples.split(",")))
    args.baselines = valid_parse(
        args.baselines, possible_values=list(BASELINE_STRS), arg_type="baselines"
    )
    args.models = valid_parse(
        args.models, possible_values=list(MODEL_STRS.keys()), arg_type="models"
    )

    baselines = lookup_string(BASELINE_STRS, args.baselines)
    # models = lookup_string(MODEL_STRS, args.models)

    return (args.examples, baselines, args.models)


# parse string into list of strings. Check that each string is an allowed value.
def valid_parse(args_str: str, possible_values: List[str], arg_type: str) -> List[str]:
    args = args_str.split(",")
    for arg in args:
        assert (
            arg in possible_values
        ), f"{arg} is an invalid value for {arg_type}. Possible values are: {possible_values}"

    return args


def lookup_string(table: Dict[str, Any], keys: List[str]) -> List[Any]:
    return [table[key] for key in keys]
