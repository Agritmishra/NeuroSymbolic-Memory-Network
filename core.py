import re
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import networkx as nx

try:
    from pyvis.network import Network 
except Exception:
    Network = None

try:
    from sentence_transformers import SentenceTransformer
    SB = True
except Exception:
    SB = False

# huggingFace transformers pipeline for optional LLM features
try:
    from transformers import pipeline
    TRANS = True
except Exception:
    TRANS = False

from store import MemoryStore


# patterns to extract subject/predicate/object
PATTERNS = [

    # Identity 
    r"(?P<L>.+?)\s+(?:is|are|was|were)\s+(?:a|an|the)?\s*(?P<R>.+)",
    r"(?P<L>.+?)\s+(?:is|are)\s+(?:called|known as)\s+(?P<R>.+)",

    # for conversion 
    r"(?P<L>.+?)\s+(?:becomes?|turns into|converted\s+to|transforms into)\s+(?P<R>.+)",
    r"(?P<L>.+?)\s+(?:changes into|evolves into|develops into)\s+(?P<R>.+)",

    # production
    r"(?P<L>.+?)\s+(?:produces?|generates?|creates?|forms?|makes?)\s+(?P<R>.+)",
    r"(?P<L>.+?)\s+(?:results? in|leads to|brings about)\s+(?P<R>.+)",

    # for causation
    r"(?P<L>.+?)\s+(?:causes?|induces?|triggers?|provokes?)\s+(?P<R>.+)",
    r"(?P<L>.+?)\s+(?:is caused by)\s+(?P<R>.+)",

    # for purpose /function
    r"(?P<L>.+?)\s+(?:is used for|is used to|serves to|serves as)\s+(?P<R>.+)",
    r"(?P<L>.+?)\s+(?:helps|helps to|helps in)\s+(?P<R>.+)",
    r"(?P<L>.+?)\s+(?:supports?|contributes to)\s+(?P<R>.+)",

    r"(?P<L>.+?)\s+(?:contains?|includes?|holds)\s+(?P<R>.+)",
    r"(?P<L>.+?)\s+(?:is part of|is a part of|is a component of|belongs to)\s+(?P<R>.+)",

    r"(?P<L>.+?)\s+(?:needs?|requires?|depends on|is dependent on)\s+(?P<R>.+)",
    r"(?P<L>.+?)\s+(?:is necessary for|is required for)\s+(?P<R>.+)",

    # essential
    r"(?P<L>.+?)\s+(?:is essential for|is crucial for|is vital for|is important for)\s+(?P<R>.+)",

    # relationship
    r"(?P<L>.+?)\s+(?:relates to|is related to|is connected to)\s+(?P<R>.+)",
    r"(?P<L>.+?)\s+(?:affects?|influences?)\s+(?P<R>.+)",

    r"(?P<L>.+?)\s+(?:has|have|possesses?)\s+(?P<R>.+)",


    r"(?P<L>.+?)\s+(?:uses?|utilizes?)\s+(?P<R>.+)",
    r"(?P<L>.+?)\s+(?:improves?|enhances?)\s+(?P<R>.+)",
]



class SymbolicMemoryNetwork:
    """
    Neuro-symbolic memory network
    """

    def __init__(
        self,
        encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        verifier_name: str = "google/flan-t5-small",
        use_llm_verify: bool = False,
        use_llm_augment: bool = False,
    ) -> None:
        self.store = MemoryStore()
        self.graph = nx.DiGraph()

        self.encoder: Optional["SentenceTransformer"] = None
        self.verifier = None
        self.augmenter = None

        self.use_llm_verify = use_llm_verify
        self.use_llm_augment = use_llm_augment

        #HuggingFace sentence encoder
        if SB:
            try:
                self.encoder = SentenceTransformer(encoder_name)
            except Exception:
                self.encoder = None

        # HuggingFace verifier/augmenter......optional
        if TRANS and (use_llm_verify or use_llm_augment):
            try:
                pipe = pipeline(
                    "text2text-generation",
                    model=verifier_name,
                    tokenizer=verifier_name,
                    device="cpu",
                )
                if use_llm_verify:
                    self.verifier = pipe
                if use_llm_augment:
                    self.augmenter = pipe
            except Exception:
                self.verifier = None
                self.augmenter = None


    #   internal function

    def _norm(self, s: Optional[str]) -> Optional[str]:
        """normalize for case insensitive graph keys."""
        if isinstance(s, str):
            return s.strip().lower()
        return s

    def _embed(self, text: str) -> Optional[np.ndarray]:
        """encode text into a vector using a HF sentence-transformers model."""
        if self.encoder is None or not text:
            return None
        try:
            v = self.encoder.encode([text], convert_to_numpy=True)
            return v[0]
        except Exception:
            return None

    def _classify_relation(self, text: str) -> str:
        t = text.lower()

        if any(x in t for x in [
            "causes", "induces", "triggers", "provokes", "leads to", "results in", "brings about"
        ]):
            return "causes"

        if any(x in t for x in [
             "produces", "generates", "creates", "forms", "makes"
        ]):
            return "produces"

        if any(x in t for x in [
            "is used for", "is used to", "serves to", "serves as"
        ]):
            return "used_for"

        if any(x in t for x in [
            "needs", "requires", "depends on", "is required for", "is necessary for"
        ]):
            return "requires"

        if any(x in t for x in [
            "is essential for", "is crucial for", "is vital for", "is important for"
        ]):
            return "essential_for"

        if any(x in t for x in [
            "contains", "includes", "holds", "is part of", "is a part of", "is a component of", "belongs to"
        ]):
            return "part_of"
 
        if any(x in t for x in [
            "supports", "contributes to", "helps", "helps to", "helps in"
        ]):
            return "supports"

        if any(x in t for x in [
            "uses", "utilizes"
        ]):
            return "uses"

        if any(x in t for x in [
            "affects", "influences"
        ]):
            return "affects"

        if any(x in t for x in [
            "is related to", "relates to", "is connected to"
        ]):
            return "related_to"

        if any(x in t for x in [
            "becomes", "turns into", "converted to", "transforms into", "changes into", "evolves into", "develops into"
        ]):
            return "transforms"

        # default
        return "related"


    def _extract_symbols(self, text: str) -> List[Dict[str, Any]]:
        """
        extract one or more symbolic triples from a sentence
        """
        out: List[Dict[str, Any]] = []
        s = (text or "").strip()

        for p in PATTERNS:
            m = re.search(p, s, flags=re.I)
            if m:
                left_raw = m.group("L").strip().strip(".")
                right_raw = m.group("R").strip().strip(".")
                left_norm = self._norm(left_raw)
                right_norm = self._norm(right_raw)
                pred = self._classify_relation(s)

                out.append(
                    {
                        "pred": pred,
                        "subj": left_norm,
                        "obj": right_norm,
                        "subj_raw": left_raw,
                        "obj_raw": right_raw,
                        "src": "pattern",
                    }
                )

        if not out and s:
            # fallback.....store the whole sentence as a statement
            out.append(
                {
                    "pred": "stmt",
                    "subj": self._norm(s),
                    "obj": None,
                    "subj_raw": s,
                    "obj_raw": None,
                    "src": "raw",
                }
            )

        return out


    #   Core memory functions

    def add_text(self, text: str, meta: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        parse text into subj,pred,obj triples..... and store them in,
        MemoryStore (with optional embeddings for retrieval)
        NetworkX graph (used later for multi-hop reasoning)
        """
        symbols = self._extract_symbols(text)
        meta = meta or {}
        ids: List[str] = []

        for sym in symbols:
            # use raw text for embedding key....better semantics
            subj_key = sym.get("subj_raw") or sym.get("subj")
            obj_key = sym.get("obj_raw") or sym.get("obj")
            key_text = f"{sym.get('pred')} {subj_key} {obj_key or ''}"

            emb = self._embed(key_text)
            item_id = self.store.add(sym, emb, meta)
            ids.append(item_id)

            # Graph nodes/edges use normalized strings for case-insensitivity
            subj = sym.get("subj")
            obj = sym.get("obj")
            pred = sym.get("pred")

            if subj:
                self.graph.add_node(subj, last_raw=sym.get("subj_raw", subj))
            if obj:
                self.graph.add_node(obj, last_raw=sym.get("obj_raw", obj))
            if subj and obj:
                self.graph.add_edge(subj, obj, pred=pred, meta=meta)

        return ids

    def retrieve(self, query: str, k: int = 6) -> List[Dict[str, Any]]:
        """
        retrieve k most similar stored facts for a query string
        using brute-force cosine similarity over embeddings.
        """
        q_emb = self._embed(query)
        if q_emb is None:
            return []

        hits = self.store.search_bruteforce(q_emb, k=k)
        out: List[Dict[str, Any]] = []
        for h in hits:
            item = h["item"]
            out.append(
                {
                    "id": item.id,
                    "symbol": item.symbol,
                    "score": round(h["score"], 4),
                    "meta": item.meta,
                }
            )
        return out

    #   Reasoning & composition

    def compose(
        self,
        retrieved: List[Dict[str, Any]],
        query: str,
        max_hops: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        candidate reasoning chains using......
        Local composition (where obj of one equals subj of another)
        Graph multi-hop paths (up to max_hops)
        Single-fact chains
        """
        chains: List[Dict[str, Any]] = []

        # 1) Local composition
        for r1 in retrieved:
            s1 = r1["symbol"]
            for r2 in retrieved:
                if r1 is r2:
                    continue
                s2 = r2["symbol"]

                obj1 = s1.get("obj")
                subj2 = s2.get("subj")
                if obj1 and subj2 and obj1 == subj2:
                    score = (r1["score"] + r2["score"]) / 2.0
                    chains.append({"chain": [s1, s2], "score": score})

        # 2. Graph multi-hop paths
        entities = set()
        for r in retrieved:
            sym = r["symbol"]
            if sym.get("subj"):
                entities.add(sym["subj"])
            if sym.get("obj"):
                entities.add(sym["obj"])

        entities = list(entities)

        for i in range(len(entities)):
            for j in range(len(entities)):
                if i == j:
                    continue

                src = entities[i]
                dst = entities[j]

                if not (self.graph.has_node(src) and self.graph.has_node(dst)):
                    continue

                try:
                    # shortest_simple_paths is a generator, robust and scalable
                    paths_gen = nx.shortest_simple_paths(self.graph, src, dst)
                    count = 0
                    for path in paths_gen:
                        if len(path) - 1 > max_hops:
                            break

                        edge_syms: List[Dict[str, Any]] = []
                        for u, v in zip(path, path[1:]):
                            data = self.graph.get_edge_data(u, v) or {}
                            pred = data.get("pred", "related")
                            edge_syms.append(
                                {
                                    "subj": u,
                                    "pred": pred,
                                    "obj": v,
                                    "src": "graph",
                                }
                            )

                        if edge_syms:
                            # Reward longer chains slightly
                            score = 0.5 + 0.4 * len(edge_syms)
                            chains.append({"chain": edge_syms, "score": score})

                        count += 1
                        if count >= 2:
                            break
                except (nx.NetworkXNoPath, nx.NodeNotFound, nx.NetworkXError):
                    continue

        # 3) single-fact fallback chains 
        for r in retrieved:
            chains.append({"chain": [r["symbol"]], "score": r["score"]})

        chains.sort(key=lambda x: x["score"], reverse=True)
        return chains


    #   Chain verification

    def verify_chain(
        self,
        chain: List[Dict[str, Any]],
        query: str,
    ) -> Tuple[float, str]:
        """
        Verification...
        confidence is purely heuristic length-based... stable and deterministic.
         If an LLM is enabled, it is used only to generate a short rationale.
        """
        if not chain:
            return 0.0, "HEURISTIC: empty chain"

        # base heuristic score: longer chains => higher confidence
        score = max(0.2, min(0.95, 0.5 + 0.1 * len(chain)))

        text = " ".join(
            f"{c.get('subj')} {c.get('pred')} {c.get('obj') or ''}"
            for c in chain
        ).strip()

        rationale = f"HEURISTIC: based on {len(chain)} step(s)"

        if self.use_llm_verify and self.verifier is not None:
            prompt = (
                "You are a reasoning assistant.\n"
                f"Facts: {text}\n"
                f"Question: {query}\n"
                "Briefly explain in one sentence how these facts relate to the question."
            )
            try:
                gen = self.verifier(
                    prompt,
                    max_length=64,
                    num_return_sequences=1,
                )
                first = gen[0] if isinstance(gen, list) else gen

                if isinstance(first, dict):
                    raw = (
                        first.get("generated_text")
                        or first.get("text")
                        or first.get("summary_text")
                        or str(first)
                    )
                else:
                    raw = str(first)

                rationale = "LLM: " + raw.strip().replace("\n", " ")[:200]
            except Exception:
                # fall back to heuristic rationale
                pass

        return float(score), rationale

    #   LLM Augmentation (optional)

    def _augment_with_llm(
        self,
        query: str,
        retrieved: List[Dict[str, Any]],
        max_new_facts: int = 3,
    ) -> List[str]:
        """
        ask a small LLM to propose bridging facts when reasoning is weak.
        """
        if (
            not self.use_llm_augment
            or self.augmenter is None
            or max_new_facts <= 0
        ):
            return []

        # build a compact context
        context_lines: List[str] = []
        for r in retrieved[:5]:
            sym = r["symbol"]
            sj = sym.get("subj_raw") or sym.get("subj") or ""
            pj = sym.get("pred") or ""
            oj = sym.get("obj_raw") or sym.get("obj") or ""
            line = f"- {sj} {pj} {oj}".strip()
            context_lines.append(line)

        context = "\n".join(context_lines) if context_lines else "No facts yet."

        prompt = (
            "You are a reasoning assistant. I will give you a question and some known facts. "
            "Propose up to "
            f"{max_new_facts} additional short facts that could help connect these facts "
            "to answer the question.\n\n"
            f"Question: {query}\n"
            f"Known facts:\n{context}\n\n"
            "Return your answer in JSON as a list under the key 'facts', where each fact has "
            "keys 'subj', 'pred', and 'obj'. Example:\n"
            '{"facts": [{"subj": "A", "pred": "causes", "obj": "B"}]}'
        )

        try:
            gen = self.augmenter(
                prompt,
                max_length=256,
                num_return_sequences=1,
            )

            first = gen[0] if isinstance(gen, list) else gen
            if isinstance(first, dict):
                raw = (
                    first.get("generated_text")
                    or first.get("text")
                    or first.get("summary_text")
                    or str(first)
                )
            else:
                raw = str(first)

            # Try to parse JSON 
            match = re.search(r"(\{[\s\S]*?\})", raw)
            new_ids: List[str] = []

            if match:
                try:
                    parsed = json.loads(match.group(1))
                    facts = parsed.get("facts", [])
                    for f in facts[:max_new_facts]:
                        subj = f.get("subj")
                        pred = f.get("pred")
                        obj = f.get("obj")
                        if subj and pred and obj:
                            text = f"{subj} {pred} {obj}"
                            ids = self.add_text(text, meta={"src": "llm_augment"})
                            new_ids.extend(ids)
                    return new_ids
                except Exception:
                    # if JSON parse fails, fall through to raw lines
                    pass

            # Fallback...treat lines as fact sentences, but filter junk
            for line in raw.split("\n"):
                line = line.lstrip("-• ").strip()
                if not line:
                    continue
                if len(line.split()) < 3:
                    continue
                ids = self.add_text(line, meta={"src": "llm_augment_raw"})
                new_ids.extend(ids)

            return new_ids
        except Exception:
            return []

    #   Answer pipeline

    def _render_chain_step(self, step: Dict[str, Any]) -> str:
        """
        Build a readable sentence from a chain step
        """
        subj = step.get("subj")
        obj = step.get("obj")
        pred = step.get("pred") or ""

        # default to empty string if None
        subj_raw = subj or ""
        obj_raw = obj or ""

        if subj and subj in self.graph.nodes:
            try:
                subj_raw = self.graph.nodes[subj].get("last_raw", subj_raw)
            except Exception:
                pass

        if obj and obj in self.graph.nodes:
            try:
                obj_raw = self.graph.nodes[obj].get("last_raw", obj_raw)
            except Exception:
                pass

        if not subj_raw or not pred or not obj_raw:
            # simple fallback
            return f"{subj or ''} {pred or ''} {obj or ''}".strip()

        p = str(pred).lower()
        if p == "performs":
            return f"{subj_raw} performs {obj_raw}"
        if p == "produces":
            return f"{subj_raw} produces {obj_raw}"
        if p == "essential_for":
            return f"{subj_raw} is essential for {obj_raw}"
        if p == "type_of":
            return f"{subj_raw} is a type of {obj_raw}"
        if p == "part_of":
            return f"{subj_raw} is part of {obj_raw}"
        if p == "causes":
            return f"{subj_raw} causes {obj_raw}"

        return f"{subj_raw} {pred} {obj_raw}"

    def answer(self, query: str, k: int = 6) -> Dict[str, Any]:
        """
        Full QA pipeline
        """
        # 1. Initial retrieval
        retrieved = self.retrieve(query, k=k)

        # 2 Compose initial chains 
        chains = self.compose(retrieved, query)

        # If nothing at all, try augmentation once (optional)
        if not chains:
            self._augment_with_llm(query, retrieved, max_new_facts=3)
            retrieved = self.retrieve(query, k=k)
            chains = self.compose(retrieved, query)
            if not chains:
                return {
                    "query": query,
                    "retrieved": retrieved,
                    "candidates": [],
                    "best_chain_text": "No facts or reasoning chains could be generated.",
                }

        # 3. Evaluate candidate chains
        candidates: List[Dict[str, Any]] = []

        import re as _re
        q_tokens = set(_re.findall(r"\w+", query.lower())) or set()

        for ch in chains[:8]:
            chain_steps = ch.get("chain", []) or []
            raw_score = float(ch.get("score", 0.0))

            # Build plain text for the chain
            chain_text = " ".join(
                f"{s.get('subj','')} {s.get('pred','')} {s.get('obj') or ''}"
                for s in chain_steps
            )
            c_tokens = set(_re.findall(r"\w+", chain_text.lower()))

            # 1) lexical overlap with query (0–1)
            overlap = 0.0
            if q_tokens:
                overlap = len(q_tokens & c_tokens) / len(q_tokens)

            # 2) length bonus (0–1, more steps => more bonus, capped)
            length_bonus = min(1.0, 0.3 * len(chain_steps))

            # 3) combine everything into a 0–1 chain score
            combined = (
                0.4 * max(0.0, min(1.0, raw_score)) +
                0.35 * overlap +
                0.25 * length_bonus
            )
            chain_score = round(min(1.0, combined), 4)

            # verification used only for explanation + a separate verify_score
            conf, rationale = self.verify_chain(chain_steps, query)

            candidates.append(
                {
                    "chain": chain_steps,
                    "chain_score": chain_score,
                    "verify_score": round(float(conf), 3),
                    "rationale": rationale,
                }
            )

        if not candidates:
            return {
                "query": query,
                "retrieved": retrieved,
                "candidates": [],
                "best_chain_text": "No candidate chains could be evaluated.",
            }

        # Prefer....longer chains → higher chain_score
        candidates.sort(
            key=lambda c: (len(c["chain"]), c["chain_score"]),
            reverse=True,
        )
        best = candidates[0]
        best_chain_steps = best.get("chain", [])

        # 4. Build readable explanation 
        sentences: List[str] = []
        for step in best_chain_steps:
            text = self._render_chain_step(step)
            if text:
                sentences.append(text)

        best_chain_text = ". ".join(sentences).strip()

        return {
            "query": query,
            "retrieved": retrieved,
            "candidates": candidates,
            "best_chain_text": best_chain_text or "No textual explanation was built.",
        }
