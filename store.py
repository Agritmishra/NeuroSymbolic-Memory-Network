import time
import uuid
import numpy as np
from typing import Any, Dict, List, Optional

class MemoryItem:
    def __init__(self, symbol: Dict[str, Any], emb: Optional[np.ndarray], meta: Dict[str, Any]):
        self.id = str(uuid.uuid4())
        self.symbol = symbol
        self.emb = emb
        self.meta = meta or {}
        self.ts = time.time()

class MemoryStore:
    def __init__(self):
        self.items: List[MemoryItem] = []
        self._emb_matrix: Optional[np.ndarray] = None

    def add(self, symbol: Dict[str, Any], emb: Optional[np.ndarray], meta: Optional[Dict[str, Any]] = None):
        item = MemoryItem(symbol, emb, meta or {})
        self.items.append(item)
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

    def search_bruteforce(self, q_emb: Optional[np.ndarray], k: int = 5):
        if q_emb is None or self._emb_matrix is None or len(self.items) == 0:
            return []
        mat = self._emb_matrix
        eps = 1e-8
        q_norm = np.linalg.norm(q_emb) + eps
        q = q_emb / q_norm
        mat_norm = np.linalg.norm(mat, axis=1, keepdims=True) + eps
        M = mat / mat_norm
        scores = (M @ q).astype(float)
        idx = (-scores).argsort()[:k]
        out = []
        for i in idx:
            out.append({"item": self.items[i], "score": float(scores[i])})
        return out
