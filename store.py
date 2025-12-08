import time
import uuid
from typing import Any, Dict, List, Optional

import numpy as np


class MemoryItem:
    """container for single stored fact and its embedding."""

    def __init__(self, symbol: Dict[str, Any], emb: Optional[np.ndarray], meta: Optional[Dict[str, Any]] = None):
        self.id = str(uuid.uuid4())
        self.symbol = symbol
        self.emb = emb
        self.meta = meta or {}
        self.ts = time.time()


class MemoryStore:
    """simple store for symbolic facts and their embeddings"""

    def __init__(self) -> None:
        self.items: List[MemoryItem] = []
        self._emb_matrix: Optional[np.ndarray] = None

    def _rebuild_emb_matrix(self) -> None:
        
        """Rebuild embedding matrix from items that have embeddings"""
        embs = [it.emb for it in self.items if it.emb is not None]
        if not embs:
            self._emb_matrix = None
        else:
            self._emb_matrix = np.vstack(embs)

    def add(self, symbol: Dict[str, Any], emb: Optional[np.ndarray], meta: Optional[Dict[str, Any]] = None) -> str:
        """add one fact and update the embedding matrix"""
        item = MemoryItem(symbol, emb, meta or {})
        self.items.append(item)

        # if any item has no embedding, rebuild when needed.
        if emb is None:
            self._emb_matrix = None
        else:
            if self._emb_matrix is None:
                self._emb_matrix = np.array([emb])
            else:
                self._emb_matrix = np.vstack([self._emb_matrix, emb])

        return item.id

    def all(self) -> List[MemoryItem]:
        return list(self.items)

    def clear(self) -> None:
        """remove all items from the store."""
        self.items = []
        self._emb_matrix = None

    def search_bruteforce(self, q_emb: Optional[np.ndarray], k: int = 5):
        """Cosine-similarity search over all items that have embeddings"""
        
        if q_emb is None or len(self.items) == 0:
            return []

        # rebuild the matrix if needed.
        if self._emb_matrix is None:
            self._rebuild_emb_matrix()

        if self._emb_matrix is None:
        
            return []

        mat = self._emb_matrix
        eps = 1e-8
        q_norm = np.linalg.norm(q_emb) + eps
        q = q_emb / q_norm
        mat_norm = np.linalg.norm(mat, axis=1, keepdims=True) + eps
        M = mat / mat_norm
        scores = (M @ q).astype(float)

        idx = (-scores).argsort()[:k]

        # only items with embeddings are represented in _emb_matrix.
        emb_items = [it for it in self.items if it.emb is not None]
        out = []
        for i in idx:
            i_int = int(i)
            out.append({"item": emb_items[i_int], "score": float(scores[i_int])})
        return out
