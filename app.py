import json
import streamlit as st
from core import SymbolicMemoryNetwork

st.set_page_config(page_title="Symbolic Memory Network", layout="wide")
st.title("Symbolic Memory Network")

@st.cache_resource
def init_core():
    # LLM are optional for stable demo
    return SymbolicMemoryNetwork(
        use_llm_verify=False,   # set True later 
        use_llm_augment=False,  # set True later 
    )


core = init_core()

col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Interaction")
    inp = st.text_area("Add fact or ask question", height=140, value="photosynthesis produces oxygen")
    add_col, ask_col = st.columns(2)
    with add_col:
        if st.button("Add to memory"):
            ids = core.add_text(inp, meta={"source": "user"})
            st.success(f"Stored {len(ids)} item(s)")
    with ask_col:
        if st.button("Ask"):
            res = core.answer(inp, k=8)
            st.markdown("### Answer")
            if res.get("best_chain_text"):
                st.write(res["best_chain_text"])
            else:
                st.write("No confident chain found.")
            st.markdown("### Candidates")
            for c in res["candidates"][:6]:
                st.markdown(f"- chain_score={c['chain_score']} · verify={c['verify_score']} · {c['rationale']}")
                for f in c["chain"]:
                    st.write(f"    • {f.get('subj')} — {f.get('pred')} — {f.get('obj')}")
            st.markdown("### Retrieved facts")
            for r in res["retrieved"][:8]:
                sym = r["symbol"]
                st.write(f"- {sym.get('subj')} | {sym.get('pred')} | {sym.get('obj')} (score {r['score']})")

with col2:
    st.subheader("Memory")
    if st.button("Show all memory"):
        items = core.store.all()
        for it in items:
            st.write(f"- {it.id} · {it.symbol.get('subj')} | {it.symbol.get('pred')} | {it.symbol.get('obj')}")
    st.markdown("---")
    st.subheader("Utilities")
    if st.button("Clear memory (restart app to persist)"):
        core.store.clear()
        core.graph.clear()
        st.info("Memory cleared.")

st.markdown("---")
st.caption(
    "This demo uses neural embeddings + symbolic graph reasoning. "
    "LLM-based explanation and augmentation are optional and disabled by default "
    "for stability, the current answers are based on the learned facts and graph structure."
)

