import pickle
import argparse
from sklearn.neighbors import kneighbors_graph
from transformers import AutoTokenizer
import torch

from src.helper_functions import get_word_embeddings
from src.parse_arguments import MODEL_STRS


def main(args):
	device = torch.device("cpu")

	print(f'Starting KNN computation..')

	tokenizer = AutoTokenizer.from_pretrained(MODEL_STRS[args.nn])
	
	word_features		= get_word_embeddings().cpu().detach().numpy()
	word_idx_map		= tokenizer.get_vocab()
	A					= kneighbors_graph(word_features, args.nbrs, mode='distance', n_jobs=args.procs)

	knn_fname = f'processed/knns/{args.nn}_{args.dataset}_{args.nbrs}.pkl'
	with open(knn_fname, 'wb') as f:
		pickle.dump([word_idx_map, word_features, A], f)

	print(f'Written KNN data at {knn_fname}')


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='knn')
	parser.add_argument('-nn',    	default='distilbert', choices=['distilbert', 'bert'], dest='nn')
	parser.add_argument('-dataset', default='sst2', choices=['sst2', 'imdb', 'rotten'])
	parser.add_argument('-procs',	default=40, type=int)
	parser.add_argument('-nbrs',  	default=500, type=int)

	args = parser.parse_args()

	main(args)
