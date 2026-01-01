📄 **Technical Report**: [Neuro-Symbolic Memory Network – PDF](./Technical_Report.pdf)

---
# 🌟 NeuroSymbolic Memory Network (NSMN)
### Explainable Multi-Hop Reasoning with Neural Embeddings + Symbolic Graph Inference

The NeuroSymbolic Memory Network (NSMN) is a hybrid reasoning system that combines neural semantic retrieval with symbolic graph-based inference. It generates transparent, multi-hop explanations from natural-language facts, providing interpretable reasoning instead of black-box answers.

---

## 🧠 Key Features

### 🔹 Triple Extraction
Natural language facts are converted into structured triples:

    (subject) — (relation) — (object)

using a curated set of linguistic patterns capturing:
causation, transformation, production, essentiality, requirements, part–whole, support, usage, identity, and association.

---

### 🔹 Neural Semantic Retrieval
Facts are embedded using Sentence Transformers (MiniLM).  
Queries retrieve relevant facts using cosine similarity, enabling semantic matching even with different wording.

---

### 🔹 Symbolic Knowledge Graph
Extracted triples form a directed graph:

    A  --relation-->  B

Using networkx, the system performs:
- multi-hop inference
- causal/structural chain discovery
- logical reasoning across facts

---

### 🔹 Multi-Hop Chain Composition
Given a question, NSMN:

1. retrieves relevant facts using embeddings  
2. explores symbolic graph paths  
3. composes multi-step reasoning chains  
4. scores each chain using semantic + structural criteria  
5. returns a clear, interpretable explanation  

All reasoning steps are fully visible.

---

### 🔹 Optional LLM Modules
LLMs are optional (disabled by default). When enabled, they can:
- provide natural-language rationales  
- suggest additional bridging facts  

The core reasoning remains symbolic and explainable.

---

## 🎮 Interactive App

A Streamlit user interface supports:

- adding new facts  
- asking questions  
- viewing reasoning chains  
- inspecting retrieved facts  
- exploring memory  
- clearing/resetting the knowledge base  

Run locally with:

    streamlit run app.py

---

# 🔍 Examples

Below are fully working examples you can paste directly into the app.

---

## ✅ Example 1 — Leaves → Sunlight Reasoning

### Facts:

    Leaves are part of a plant
    Leaves support photosynthesis
    Photosynthesis requires sunlight

### Query:

    How are leaves connected to sunlight?

---

## ✅ Example 2 — Ice → Turbine Rotation Reasoning

### Facts:

    Ice turns into water
    Water turns into steam
    Steam causes pressure increase
    Pressure increase leads to turbine rotation

### Query:

    How does ice lead to turbine rotation?

---

# 🧱 Architecture
                          ┌──────────────────────────────┐
                          │     Natural Language Input    │
                          │ (User facts & questions)      │
                          └───────────────┬──────────────┘
                                          ▼
                          ┌──────────────────────────────┐
                          │       Triple Extraction       │
                          │ (pattern-based NLP → triples) │
                          └───────────────┬──────────────┘
                                          ▼
              ┌────────────────────────────────────────────────────────┐
              │                 Symbolic Memory Store                  │
              │  - stores triples (subj, pred, obj)                    │
              │  - creates neural embeddings (SentenceTransformer)     │
              │  - supports cosine similarity retrieval                │
              └───────────────┬───────────────────────────────┬───────┘
                              │                               │
                              ▼                               ▼
              ┌──────────────────────────────┐   ┌──────────────────────────────┐
              │      Knowledge Graph         │   │        Neural Retrieval       │
              │ (networkx DiGraph of triples)│   │ (bruteforce cosine search)    │
              └───────────────┬──────────────┘   └──────────────┬──────────────┘
                              │                                 │
                              └───────────────┬─────────────────┘
                                              ▼
                          ┌──────────────────────────────┐
                          │        Chain Composer         │
                          │  - local composition          │
                          │    (object → subject links)   │
                          │  - graph multi-hop paths      │
                          │  - generates reasoning chains │
                          └───────────────┬──────────────┘
                                          ▼
                          ┌──────────────────────────────┐
                          │     Hybrid Scoring Engine     │
                          │  - semantic similarity        │
                          │  - lexical overlap            │
                          │  - chain depth bonus          │
                          └───────────────┬──────────────┘
                                          ▼
                          ┌──────────────────────────────┐
                          │   Final Explainable Output    │
                          │ (step-by-step reasoning chain)│
                          └──────────────────────────────┘

---

# 🏛 Why This Project Matters

NSMN demonstrates:

- hybrid neural–symbolic architecture  
- interpretable, step-by-step reasoning  
- multi-hop logical chains  
- semantic retrieval with embeddings  
- graph-based inference  
- optional LLM enhancement without dependency  
- a clean deployment via Streamlit  

---

# 🔮 Future Directions

- interactive knowledge-graph visualization  
- FAISS / ScaNN accelerated retrieval  
- transformer-based relation extraction  
- differentiable reasoning modules  
- formal evaluations on chain-depth tasks  
- integration with retrieval-augmented LLMs  

---

# 📄 Citation

    Agrit Mishra. "NeuroSymbolic Memory Network: 
    Explainable Multi-Hop Reasoning with Hybrid Neural-Symbolic Architecture." 2025.

---

# 🎓 License

MIT License.

