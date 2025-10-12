import re
import json
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import networkx as nx
from pyvis.network import Network

try:
    from sentence_transformers import SentenceTransformer
    SB = True
except Exception:
    SB = False

try:
    from transformers import pipeline
    TRANS = True
except Exception:
    TRANS = False

from store import MemoryStore

PATTERNS = [
    r"(?P<L>.+?)\b(?:is a|is an|is the|are|was|were|becomes?)\b(?P<R>.+)",
    r"(?P<L>.+?)\b(?:has|have|contains|includes)\b(?P<R>.+)",
    r"(?P<L>.+?)\b(?:produces|generates|creates|gives)\b(?P<R>.+)",
]

class SymbolicMemoryNetwork:
    def __init__(self, encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2", verifier_name: str = "google/flan-t5-base"):
        self.store = MemoryStore()
        self.encoder = None
        self.verifier = None
        if SB:
            try:
                self.encoder = SentenceTransformer(encoder_name)
            except Exception:
                self.encoder = None
        if TRANS:
            try:
                self.verifier = pipeline("text2text-generation", model=verifier_name, tokenizer=verifier_name, device=-1)
            except Exception:
                self.verifier = None

    def _embed(self, text: str) -> Optional[np.ndarray]:
        if self.encoder is None or not text:
            return None
        try:
            v = self.encoder.encode([text], convert_to_numpy=True)
            return v[0]
        except Exception:
            return None

    def _extract_symbols(self, text: str) -> List[Dict[str, Any]]:
        out = []
        s = (text or "").strip()
        for p in PATTERNS:
            m = re.search(p, s, flags=re.I)
            if m:
                L = m.group("L").strip().strip(".")
                R = m.group("R").strip().strip(".")
                out.append({"pred": "related", "subj": L, "obj": R, "src": "pattern"})
        if not out and len(s) > 0:
            out.append({"pred": "stmt", "subj": s, "obj": None, "src": "raw"})
        return out

    def add_text(self, text: str, meta: Optional[Dict[str, Any]] = None) -> List[str]:
        symbols = self._extract_symbols(text)
        ids = []
        for sym in symbols:
            key_text = f"{sym.get('pred')} {sym.get('subj')} {sym.get('obj') or ''}"
            emb = self._embed(key_text)
            ids.append(self.store.add(sym, emb, meta or {}))
        return ids

    def retrieve(self, query: str, k: int = 6) -> List[Dict[str, Any]]:
        q_emb = self._embed(query)
        hits = self.store.search_bruteforce(q_emb, k=k)
        out = []
        for h in hits:
            itm = h["item"]
            out.append({"id": itm.id, "symbol": itm.symbol, "score": round(h["score"], 4), "meta": itm.meta})
        return out

    def compose(self, retrieved: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        chains = []
        for r in retrieved:
            s = r["symbol"]
            for r2 in retrieved:
                if r is r2:
                    continue
                s2 = r2["symbol"]
                if s.get("subj") and s2.get("obj") and s.get("subj").lower() == (s2.get("obj") or "").lower():
                    chains.append({"chain": [s2, s], "score": (r["score"] + r2["score"]) / 2})
        for r in retrieved:
            chains.append({"chain": [r["symbol"]], "score": r["score"]})
        chains.sort(key=lambda x: x["score"], reverse=True)
        return chains

    def verify_chain(self, chain: List[Dict[str, Any]], query: str) -> Tuple[float, str]:
        text = " ".join([f"{c.get('subj')} {c.get('pred')} {c.get('obj') or ''}" for c in chain])
        prompt = f"Context facts: {text}\nQuestion: Is it correct to conclude '{query}' based on these facts? Answer yes/no and give one-line reason. Return JSON."
        if self.verifier is not None:
            try:
                gen = self.verifier(prompt, max_length=128, temperature=0.0, num_return_sequences=1)
                raw = gen[0]["generated_text"] if isinstance(gen, list) else str(gen)
                jmatch = re.search(r"(\{[\s\S]*\})", raw)
                if jmatch:
                    try:
                        parsed = json.loads(jmatch.group(1))
                        ans = parsed.get("answer", "")
                        conf = parsed.get("confidence", None)
                        if conf is None:
                            conf = 1.0 if "yes" in ans.lower() else 0.0
                        else:
                            conf = float(conf)
                        expl = parsed.get("explain", parsed.get("explanation", ""))
                        return float(conf), str(expl)[:400]
                    except Exception:
                        pass
                if "yes" in raw.lower():
                    return 0.9, raw.strip().replace("\n", " ")[:400]
                return 0.1, raw.strip().replace("\n", " ")[:400]
            except Exception:
                pass
        score = max(0.2, min(0.95, 0.5 + 0.1 * len(chain)))
        return float(score), "heuristic"

    def answer(self, query: str, k: int = 6) -> Dict[str, Any]:
        retrieved = self.retrieve(query, k=k)
        chains = self.compose(retrieved, query)
        verified = []
        for ch in chains[:8]:
            conf, rationale = self.verify_chain(ch["chain"], query)
            verified.append({"chain": ch["chain"], "chain_score": round(ch["score"], 4), "verify_score": round(conf, 3), "rationale": rationale})
        verified.sort(key=lambda x: x["verify_score"], reverse=True)
        best = verified[0] if verified else None
        text = ""
        if best:
            parts = []
            for c in best["chain"]:
                subj = c.get("subj")
                pred = c.get("pred")
                obj = c.get("obj")
                if obj:
                    parts.append(f"{subj} {pred} {obj}")
                else:
                    parts.append(f"{subj} ({pred})")
            text = ". ".join(parts)
        return {"query": query, "retrieved": retrieved, "candidates": verified, "best_chain_text": text}
