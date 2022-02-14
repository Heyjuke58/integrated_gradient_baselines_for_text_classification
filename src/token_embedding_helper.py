from typing import Dict, List, Tuple
from src.helper_functions import get_word_embeddings, hash_tensor
from transformers import AutoModelForSequenceClassification
import torch
from torch import Tensor
from queue import PriorityQueue


class TokenEmbeddingHelper:
    """
    Helper class to get mapping from token ids to embeddings, and embeddings to token ids
    """

    def __init__(self, model: AutoModelForSequenceClassification, model_str: str) -> None:
        self.model = model
        self.model_str = model_str
        self.token_to_emb = {i: emb for i, emb in enumerate(get_word_embeddings(model, model_str))}
        self.emb_to_token = {hash_tensor(emb): i for i, emb in self.token_to_emb.items()}

    def get_token_id(self, embedding: Tensor) -> int:
        return self.emb_to_token[hash_tensor(embedding)]

    def get_emb(self, token_id: int) -> Tensor:
        return self.token_to_emb[token_id]

    def get_closest_by_token_embed_for_embed(self, embedding: Tensor) -> Tensor:
        """
        Get closest by token id with corresponding embedding for an arbitrary input embedding.

        :param embedding: Input embedding to get the closest by token id and embedding for
        """
        assert (
            embedding.shape == get_word_embeddings(self.model, self.model_str)[0].shape
        ), "Embedding tensor has invalid shape to get closest by real token embedding!"
        dists = torch.cdist(embedding.unsqueeze(0), get_word_embeddings(self.model, self.model_str))
        return self.get_emb(torch.argmin(dists).item())

    def get_topk_closest_by_token_embed_for_embed(
        self, topk: int, embedding: Tensor, tokenizer
    ) -> List[Tuple[int, str, float, Tensor]]:
        """
        Get top k closest by token id with corresponding embedding for an arbitrary input embedding.

        :param embedding: Input embedding to get the closest by token id and embedding for
        """
        assert (
            embedding.shape == get_word_embeddings(self.model, self.model_str)[0].shape
        ), "Embedding tensor has invalid shape to get closest by real token embedding!"
        pq = PriorityQueue(maxsize=topk)
        with torch.no_grad():
            for i, word_embed in enumerate(get_word_embeddings(self.model, self.model_str)):
                dist = torch.sqrt(torch.sum((word_embed - embedding) ** 2)).item()
                if not pq.full():
                    # negative priority to get the largest distance when calling pq.get()
                    pq.put((-dist, i))
                else:
                    cur_max_dist, _ = pq.queue[0]
                    if -cur_max_dist > dist:
                        pq.get()
                        pq.put((-dist, i))
            return [
                (token_id, tokenizer.convert_ids_to_tokens(token_id), -dist, self.get_emb(token_id))
                for dist, token_id in sorted(pq.queue, key=lambda tup: tup[0], reverse=True)
            ]
