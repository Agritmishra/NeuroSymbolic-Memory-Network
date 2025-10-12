# 🧠 Symbolic Memory Network

*A Hybrid Neuro-Symbolic Reasoning Engine for Knowledge Storage, Retrieval, and Logical Inference*

---

## Overview

The **Symbolic Memory Network (SMN)** is an experimental **neuro-symbolic reasoning framework** that combines **symbolic graph-based memory** with **neural embeddings** and **LLM-based verification** to simulate structured reasoning.

It allows a user to:

* Store natural language facts into an *interpretable symbolic memory graph*
* Retrieve semantically similar knowledge using vector search
* Compose and verify reasoning chains using a hybrid **neural + symbolic inference loop**

This project demonstrates how large language models can be augmented with **structured memory**, **semantic embeddings**, and **symbolic logic patterns** — a crucial research direction for building *explainable and trustworthy AI systems*.

---

## Architecture Overview

### Core Components

#### 1. **Symbol Extraction Layer**

* Converts raw text into structured symbolic triples (`subject`, `predicate`, `object`).
* Uses lightweight regex-based pattern recognition (`is a`, `has`, `produces`, etc.).
* Falls back to “statement” nodes for unstructured input.

#### 2. **Neural Embedding Layer**

* Uses `SentenceTransformer` (MiniLM-L6-v2) to embed facts and queries into vector space.
* Enables **semantic retrieval** based on conceptual similarity.
* Supports both CPU-only and optional GPU acceleration.

#### 3. **Memory Store**

* A minimal persistent memory engine (`store.py`) for storing:

  * Symbolic items
  * Vector embeddings
  * Metadata (source, timestamp, etc.)
* Provides **brute-force similarity search** via normalized cosine similarity.

#### 4. **Retriever and Composer**

* Retrieves top-*k* relevant memories for a given query.
* Composes *chains of reasoning* where facts share overlapping entities (`subj == obj`).
* Scores chains using average similarity and ranks them.

#### 5. **Verifier (LLM Reasoner)**

* Uses `google/flan-t5-base` (through `transformers` pipeline) to verify whether a reasoning chain supports the user’s query.
* Produces structured JSON responses (`{"answer": ..., "confidence": ..., "explanation": ...}`) when available.
* Falls back to heuristic scoring when the verifier is unavailable.

#### 6. **Frontend (Streamlit UI)**

* Interactive dashboard to:

  * Add knowledge facts
  * Query the system for reasoning
  * Visualize retrieved chains and memory contents
* Designed for interpretability and iterative experimentation.

---

## Data Flow

```text
┌───────────────────────┐
│   User Input (Fact or  │
│   Question in English) │
└──────────────────────┌┘
             │
             ▼
     Symbol Extraction
 (Regex + Shallow Parsing)
             │
             ▼
   Neural Embedding (SBERT)
             │
             ▼
  MemoryStore.add(symbol, emb)
             │
             ▼
  ┌───────────────────────────────────────────────┐
  │  Query: Retrieve &     │
  │  Compose Reason Chains  │
  └───────────────────────────────────────────────┘
             │
             ▼
     LLM Verifier (Flan-T5)
     ⟷ Confidence + Rationale
             │
             ▼
  Streamlit UI: Visualization
```

---

## 🧠 Example Interaction

| Action          | Description                                                                                         |
| --------------- | --------------------------------------------------------------------------------------------------- |
| **Add Fact**    | “Photosynthesis produces oxygen” → stored as symbolic relation `(photosynthesis, produces, oxygen)` |
| **Ask Query**   | “Does photosynthesis generate oxygen?”                                                              |
| **SMN Process** | Retrieves similar embeddings → finds reasoning chain → verifies via Flan-T5                         |
| **Output**      | “Yes. Because photosynthesis produces oxygen.” (Confidence: 0.92)                                   |

---

## ⚙️ Technologies Used

| Layer               | Library / Tool                         | Purpose                                |
| ------------------- | -------------------------------------- | -------------------------------------- |
| **Frontend**        | `Streamlit`                            | Interactive dashboard                  |
| **Embedding**       | `Sentence-Transformers (MiniLM-L6-v2)` | Semantic encoding of text              |
| **Verification**    | `Transformers (Flan-T5)`               | Logical verification and justification |
| **Memory**          | `Numpy`, `Faiss` (optional)            | Efficient similarity search            |
| **Knowledge Graph** | `NetworkX`, `PyVis`                    | Symbolic relation visualization        |
| **Core Logic**      | Python (OOP)                           | Modular architecture for extensibility |

---

## 🗂 Repository Structure

```
📁 Symbolic-Memory-Network/
├─ app.py                  # Streamlit frontend
├─ core.py                 # Core reasoning engine
├─ store.py                # Memory store implementation
├─ requirements.txt        # Python dependencies
└─ README.md               
```

---

## Key Innovations

1. **Hybrid Neuro-Symbolic Reasoning**

   * Merges dense embeddings with symbolic pattern extraction.
2. **Explainable Inference**

   * Every reasoning chain is human-readable.
3. **Lightweight Cognitive Architecture**

   * Runs entirely on CPU with optional neural modules.
4. **Modular Design**

   * Encoder, verifier, and retriever are pluggable.
5. **Self-Contained Memory**

   * Local, interpretable knowledge representation without a database.

---


### 3. Add Facts and Ask Questions

* Type sentences like:

  * “Water is a liquid.”
  * “Liquid has molecules.”
  * “Do molecules exist in water?”
* Observe how the system reasons via retrieved symbolic chains.

---

## 🧩 Future Directions

* **Integration with RAG pipelines** for long-context retrieval.
* **Knowledge Graph visualization panel** for reasoning trace.
* **Local fine-tuning of verifier** for domain-specific logic.
* **Temporal reasoning and episodic memory modules.**

---

## 🎓 Research Context

This project represents an early attempt toward **autonomous reasoning systems** that blend symbolic structure and neural understanding.
Such hybrid systems aim to overcome the limitations of:

* Purely statistical models (which lack explainability)
* Purely symbolic systems (which lack generalization)

The **Symbolic Memory Network** aligns with active research themes in:

* **Explainable AI (XAI)**
* **Neuro-Symbolic Integration**
* **Cognitive Architectures**
* **Interpretable Machine Reasoning**

