import argparse
from typing import Any, Dict, List, Tuple

MODEL_STRS = {
    # https://huggingface.co/textattack/bert-base-uncased-SST-2
    "bert": "textattack/bert-base-uncased-SST-2",
    # https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english
    "distilbert": "distilbert-base-uncased-finetuned-sst-2-english",
    # "sshleifer": "sshleifer/tiny-distilbert-base-uncased-finetuned-sst-2-english",
}
BASELINE_STRS = {
    "pad_embed",
    "zero_embed",
    "furthest_embed",
    "blurred_embed",
    "flipped_blurred_embed",
    "both_blurred_embed",
    "uniform",
    "gaussian",
    "furthest_word",
    "avg_word_embed",
    "avg_word",
}


def parse_arguments() -> Tuple[Any, ...]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--examples",
        type=str,
        dest="examples",
        default="0,1,2",
        help="Example indices used for IG evaluation. Can either be specified by ints seperated by ',' for explicit examples or be a range i.e. the first 1000 examples: -e \"0-1000\" ",
    )
    parser.add_argument(
        "-b",
        "--baselines",
        type=str,
        dest="baselines",
        default="furthest_embed,pad_embed,zero_embed,blurred_embed,flipped_blurred_embed,both_blurred_embed,uniform,gaussian",
        # default="blurred_embed,flipped_blurred_embed,both_blurred_embed",
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
    parser.add_argument(
        "-v",
        "--version-ig",
        type=str,
        dest="version_ig",
        default="ig",
        help="Which version auf IG should be used (vanilla 'ig' or discretized 'dig').",
    )
    parser.add_argument(
        "--dig-strategy",
        type=str,
        dest="dig_strategy",
        default="greedy",
        help="Which strategy is used for DIG (one of ['greedy', 'maxcount']).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        dest="steps",
        default=32,
        help="Number of interpolation steps for IG.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        dest="seed",
        default=33,  # super max
        help="Seed to be used for baselines that use a randomizer.",
    )
    parser.add_argument(
        "--no-viz-attr",
        dest="viz_attr",
        action="store_false",
        help="Whether to visualize the summed attributions for every example.",
    )
    parser.set_defaults(viz_attr=True)
    parser.add_argument(
        "--no-viz-topk",
        dest="viz_topk",
        action="store_false",
        help="Whether to visualize the comprehensiveness and log-odds for different k values of attributions over all examples.",
    )
    parser.set_defaults(viz_topk=True)
    parser.add_argument(
        "--viz-word-path",
        dest="viz_word_path",
        action="store_true",
        help="Whether to visualize the discrete word paths from informationless baseline to actual input.",
    )
    parser.set_defaults(viz_word_path=False)

    args = parser.parse_args()
    if "-" in args.examples:
        assert "," not in args.examples
        start, end = args.examples.split("-")
        args.examples = list(range(int(start), int(end)))
    else:
        args.examples = list(map(int, args.examples.split(",")))

    args.baselines = valid_parse(
        args.baselines, possible_values=list(BASELINE_STRS), arg_type="baselines"
    )
    args.models = valid_parse(
        args.models, possible_values=list(MODEL_STRS.keys()), arg_type="models"
    )

    # baselines = lookup_string(BASELINE_STRS, args.baselines)
    # models = lookup_string(MODEL_STRS, args.models)

    return (
        args.examples,
        args.baselines,
        args.models,
        args.version_ig,
        args.dig_strategy,
        args.steps,
        args.seed,
        args.viz_attr,
        args.viz_topk,
        args.viz_word_path,
    )


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
