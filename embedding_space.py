# check out the embeddings that are used by the models

from helper_functions import get_word_embeddings
from visualization import embedding_histogram


def main():

    # plot histogram
    all_word_embeddings = get_word_embeddings(model, model_str)
    embedding_histogram(all_word_embeddings)
    continue


if __name__ == "__main__":
    main()
