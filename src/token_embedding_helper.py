from typing import Dict
from src.helper_functions import get_word_embeddings, hash_tensor
from transformers import AutoModelForSequenceClassification
import torch
from torch import Tensor


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
        min_dist = float("inf")
        token_id: int = None
        with torch.no_grad():
            for i, word_embed in enumerate(get_word_embeddings(self.model, self.model_str)):
                dist = torch.sqrt(torch.sum((word_embed - embedding) ** 2))
                # dist = torch.cdist(word_embed, embedding) ** 2
                if dist < min_dist:
                    min_dist = dist
                    token_id = i
            return self.get_emb(token_id)
