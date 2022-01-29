import argparse
from typing import List

MODEL_STRS = {
    "distil" : "distilbert-base-uncased-finetuned-sst-2-english",
    "sshleifer": "sshleifer/tiny-distilbert-base-uncased-finetuned-sst-2-english",
}
BASELINE_STRS = {
    "todo": "TODO",
}

def parse_arguments():
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
        default="TODO",
        help="Type of baseline to be used for Integrated Gradients.",
    )
    parser.add_argument(
        "-m",
        "--models",
        type=str,
        dest="models",
        default="distill,sshleifer",
        help="Models that should be interpreted.",
    )

    args = parser.parse_args()
    args.examples = list(map(int, args.examples.split(",")))
    args.baselines = valid_parse(args.baselines, possible_values=list(BASELINE_STRS.keys()), arg_type="baselines")
    args.models = valid_parse(args.models, possible_values=list(MODEL_STRS.keys()), arg_type="models")

    return (args.examples, args.baselines, args.models)

# parse string into list of strings. Check that each string is an allowed value.
def valid_parse(args_str: str, possible_values: List[str], arg_type: str) -> List[str]:
    args = args_str.split(",")
    for arg in args:
        assert arg in possible_values, f"{arg} is an invalid value for {arg_type}. Possible values are: {possible_values}"
        
    return args
