import argparse
import pickle

from sklearn.neighbors import kneighbors_graph
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.helper_functions import get_word_embeddings
from src.parse_arguments import MODEL_STRS
from pathlib import Path


def main(args):

    print(f"Starting KNN computation..")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_STRS[args.model], return_dict=False
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_STRS[args.model])

    word_features = get_word_embeddings(model, args.model).cpu().detach().numpy()
    word_idx_map = tokenizer.get_vocab()
    A = kneighbors_graph(word_features, args.neighbors, mode="distance", n_jobs=args.processes)

    Path("knn").mkdir(parents=True, exist_ok=True)
    knn_fname = f"knn/{args.model}_{args.neighbors}.pkl"
    with open(knn_fname, "wb") as f:
        pickle.dump([word_idx_map, word_features, A], f)

    print(f"Written KNN data at {knn_fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="knn")
    parser.add_argument("-m", default="distilbert", choices=["distilbert", "bert"], dest="model")
    parser.add_argument("--processes", default=40, type=int, dest="processes")
    parser.add_argument("--neighbors", default=500, type=int, dest="neighbors")

    args = parser.parse_args()

    main(args)
