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
    "uniform",
    "gaussian",
    "blurred_embed",
    "flipped_blurred_embed",
    "both_blurred_embed",
    "avg_word_embed",
    "avg_word",
    "furthest_embed",
    "furthest_word",
    "zero_embed",
}
DISCRETE_BASELINES = {"pad_embed", "furthest_word", "avg_word"}


def parse_arguments() -> Tuple[Any, ...]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--examples",
        type=str,
        dest="examples",
        default="0",
        help="Example indices used for IG evaluation. Can either be specified by ints seperated by ',' for explicit examples or be a range i.e. the first 100 examples: -e \"0-100\" ",
    )
    parser.add_argument(
        "-b",
        "--baselines",
        type=str,
        dest="baselines",
        default="pad_embed,uniform,gaussian,blurred_embed,flipped_blurred_embed,both_blurred_embed,avg_word_embed,avg_word,furthest_embed,furthest_word,zero_embed",
        # default="blurred_embed,flipped_blurred_embed,both_blurred_embed",
        help=f"Type of baseline to be used for Integrated Gradients or Discretized Integrated Gradients. Allowed values: {BASELINE_STRS}",
    )
    parser.add_argument(
        "-m",
        "--models",
        type=str,
        dest="models",
        default="distilbert,bert",
        help=f"Models that should be interpreted. Allowed values: {set(MODEL_STRS.keys())}",
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
        default=10,
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
    parser.add_argument(
        "--viz-emb-space",
        dest="viz_emb_space",
        action="store_true",
        help="Whether to visualize the embedding space of all word embeddings using a PCA.",
    )
    parser.set_defaults(viz_emb_space=False)

    args = parser.parse_args()
    if "-" in args.examples:
        assert "," not in args.examples
        start, end = args.examples.split("-")
        args.examples = list(range(int(start), int(end)))
    else:
        args.examples = list(map(int, args.examples.split(",")))

    args.baselines = valid_parse(
        args.baselines,
        possible_values=list(BASELINE_STRS)
        if args.version_ig == "ig"
        else list(DISCRETE_BASELINES),
        arg_type="baselines",
    )
    args.models = valid_parse(
        args.models, possible_values=list(MODEL_STRS.keys()), arg_type="models"
    )

    # viz topk is for use with many samples:
    if args.viz_topk:
        assert not (
            args.viz_attr or args.viz_word_path or args.viz_emb_space
        ), "visualizations are not compatible with each other"
        assert len(args.examples) >= 50, "top-k analysis is a quantitative measure (need n >= 50!)"

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
        args.viz_emb_space,
    )


# parse string into list of strings. Check that each string is an allowed value.
def valid_parse(args_str: str, possible_values: List[str], arg_type: str) -> List[str]:
    args = args_str.split(",")
    for arg in args:
        assert (
            arg in possible_values
        ), f"{arg} is an invalid value for {arg_type}. Possible values are: {possible_values}"

    return args
