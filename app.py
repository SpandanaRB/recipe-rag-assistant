"""
app.py — Recipe RAG Assistant
Streamlit front-end for a Document Q&A RAG agent that acts as an intelligent
recipe generator powered by Groq + FAISS + sentence-transformers.
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Load .env before any other import that might need the key
load_dotenv()

import rag_utils as rag
import llm_utils as llm

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Recipe RAG Assistant",
    page_icon="🍳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        /* Chat bubbles */
        .stChatMessage { border-radius: 12px; }
        /* Sidebar section headers */
        .sidebar-section { font-weight: 600; margin-top: 0.75rem; color: #3b82d4; }
        /* Source badge */
        .source-badge {
            display: inline-block;
            background: #f0f4ff;
            border: 1px solid #c7d7f9;
            border-radius: 999px;
            padding: 2px 10px;
            font-size: 0.78rem;
            color: #3b52a0;
            margin: 2px 3px;
        }
        /* Status bar */
        .status-ok { color: #16a34a; font-size: 0.85rem; }
        .status-warn { color: #d97706; font-size: 0.85rem; }
        .status-err { color: #dc2626; font-size: 0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state initialisation ──────────────────────────────────────────────
def init_state() -> None:
    defaults = {
        "faiss_index": None,
        "chunks": [],
        "chat_history": [],          # [{"role": "user"|"assistant", "content": str}]
        "loaded_sources": set(),     # track which files are already indexed
        "sample_loaded": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_state()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🍳 Recipe RAG")
    st.caption("Intelligent recipe assistant powered by RAG + Groq")

    # ── API key status
    if llm.check_api_key():
        st.markdown('<p class="status-ok">✅ Groq API key detected</p>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<p class="status-err">⚠️ GROQ_API_KEY not set — add it to <code>.env</code> '
            "or export it as an environment variable.</p>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Document ingestion
    st.markdown('<p class="sidebar-section">📂 Recipe Documents</p>', unsafe_allow_html=True)

    # Load sample recipes
    col1, col2 = st.columns([3, 1])
    with col1:
        sample_btn = st.button(
            "Load Sample Recipes (20)",
            use_container_width=True,
            disabled=st.session_state["sample_loaded"],
        )
    with col2:
        if st.session_state["sample_loaded"]:
            st.markdown("✅", unsafe_allow_html=True)

    if sample_btn and not st.session_state["sample_loaded"]:
        with st.spinner("Embedding sample recipes…"):
            try:
                index, chunks = rag.load_sample_data()
                if index is None:
                    st.error("sample_data/ folder is empty or not found.")
                else:
                    st.session_state["faiss_index"] = index
                    st.session_state["chunks"] = chunks
                    st.session_state["sample_loaded"] = True
                    st.session_state["loaded_sources"] = set(rag.unique_sources(chunks))
                    st.success(f"Loaded {len(chunks)} chunks from {len(st.session_state['loaded_sources'])} recipes.")
            except Exception as exc:
                st.error(f"Failed to load sample data: {exc}")

    # Upload custom files
    st.markdown("**Upload your own recipes:**")
    uploaded_files = st.file_uploader(
        "PDF or .txt files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        new_files = [f for f in uploaded_files if f.name not in st.session_state["loaded_sources"]]
        if new_files:
            if st.button(f"Index {len(new_files)} new file(s)", use_container_width=True):
                with st.spinner("Chunking and embedding…"):
                    try:
                        new_chunks: list = []
                        for uf in new_files:
                            text = rag.extract_text(uf.name, uf.read())
                            source_name = Path(uf.name).stem.replace("_", " ").title()
                            chunks = rag.chunk_text(text, source_name=source_name)
                            new_chunks.extend(chunks)

                        idx, all_chunks = rag.add_to_faiss_index(
                            st.session_state["faiss_index"],
                            st.session_state["chunks"],
                            new_chunks,
                        )
                        st.session_state["faiss_index"] = idx
                        st.session_state["chunks"] = all_chunks
                        for uf in new_files:
                            st.session_state["loaded_sources"].add(uf.name)
                        st.success(f"Added {len(new_chunks)} chunks from {len(new_files)} file(s).")
                    except Exception as exc:
                        st.error(f"Indexing failed: {exc}")
        else:
            st.info("All uploaded files are already indexed.")

    # Index status
    n_chunks = len(st.session_state["chunks"])
    n_sources = len(st.session_state["loaded_sources"])
    if n_chunks:
        st.markdown(
            f'<p class="status-ok">🗂 {n_chunks} chunks · {n_sources} source(s) indexed</p>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<p class="status-warn">⚠️ No documents indexed — load samples or upload files.</p>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Personalization
    st.markdown('<p class="sidebar-section">🎛️ Personalization</p>', unsafe_allow_html=True)

    dietary_options = [
        "None", "Vegan", "Vegetarian", "Gluten-Free", "Dairy-Free",
        "Diabetic-Friendly", "Keto", "Paleo", "Nut-Free", "Low-Sodium",
    ]
    dietary_sel = st.multiselect(
        "Dietary restrictions",
        options=dietary_options,
        default=[],
        placeholder="Select restrictions…",
    )
    dietary_str = ", ".join(dietary_sel) if dietary_sel else "None"

    avail_ingredients = st.text_area(
        "Ingredients I already have",
        placeholder="e.g. eggs, butter, flour, sugar, milk",
        height=90,
    )

    max_time = st.slider(
        "Max cooking time (minutes)",
        min_value=5,
        max_value=180,
        value=60,
        step=5,
    )

    cuisine_options = [
        "Any", "Italian", "Indian", "Mexican", "Thai", "Greek",
        "Japanese", "French", "Mediterranean", "American", "Chinese",
    ]
    cuisine_pref = st.selectbox("Preferred cuisine", cuisine_options, index=0)

    preferences = {
        "dietary_restrictions": dietary_str,
        "available_ingredients": avail_ingredients.strip(),
        "max_cook_time": max_time,
        "cuisine_preference": cuisine_pref,
    }

    st.divider()

    # Retrieval settings
    st.markdown('<p class="sidebar-section">⚙️ Retrieval Settings</p>', unsafe_allow_html=True)
    top_k = st.slider("Top-K retrieved chunks", min_value=1, max_value=10, value=5)

    st.divider()

    # Clear chat
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state["chat_history"] = []
        st.rerun()


# ── Main panel ────────────────────────────────────────────────────────────────
st.title("🍳 Recipe RAG Assistant")
st.caption(
    "Ask any recipe question — I'll retrieve relevant context from your indexed "
    "recipe documents and generate a personalised answer."
)

# ── Render existing chat history ──────────────────────────────────────────────
for turn in st.session_state["chat_history"]:
    with st.chat_message(turn["role"]):
        if turn["role"] == "assistant" and turn.get("sources"):
            src_badges = "".join(
                f'<span class="source-badge">📄 {s}</span>' for s in turn["sources"]
            )
            st.markdown(
                f'<div style="margin-bottom:8px">🔍 <strong>Sources:</strong> {src_badges}</div>',
                unsafe_allow_html=True,
            )
        st.markdown(turn["content"])

# ── Welcome message when empty ────────────────────────────────────────────────
if not st.session_state["chat_history"]:
    with st.chat_message("assistant"):
        st.markdown(
            """
Welcome! 👋 I'm your **Recipe RAG Assistant**.

**To get started:**
1. Click **Load Sample Recipes** in the sidebar (or upload your own recipe files).
2. Set your dietary preferences and available ingredients.
3. Ask me anything — for example:
   - *"How do I make a vegan chocolate cake?"*
   - *"Suggest a quick gluten-free dinner under 30 minutes."*
   - *"What can I cook with eggs, spinach, and feta?"*
   - *"Give me a diabetic-friendly breakfast idea."*
            """
        )

# ── Chat input ────────────────────────────────────────────────────────────────
user_query = st.chat_input("Ask a recipe question…")

if user_query:
    # Guard: API key
    if not llm.check_api_key():
        st.error(
            "⚠️ **GROQ_API_KEY is not set.** Add it to your `.env` file:\n\n"
            "```\nGROQ_API_KEY=your_key_here\n```\nThen restart the app."
        )
        st.stop()

    # Guard: empty vector store
    if st.session_state["faiss_index"] is None or not st.session_state["chunks"]:
        st.warning(
            "⚠️ No recipe documents are indexed yet. "
            "Click **Load Sample Recipes** in the sidebar or upload your own files."
        )
        st.stop()

    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(user_query)

    # Append user turn to history
    st.session_state["chat_history"].append({"role": "user", "content": user_query})

    # Retrieval
    with st.spinner("Searching recipe index…"):
        retrieved = rag.retrieve(
            query=user_query,
            index=st.session_state["faiss_index"],
            chunks=st.session_state["chunks"],
            top_k=top_k,
        )
    sources = rag.unique_sources(retrieved)

    # LLM generation
    with st.chat_message("assistant"):
        # Source badges
        src_badges = "".join(f'<span class="source-badge">📄 {s}</span>' for s in sources)
        st.markdown(
            f'<div style="margin-bottom:8px">🔍 <strong>Sources:</strong> {src_badges}</div>',
            unsafe_allow_html=True,
        )

        with st.spinner("Generating personalised recipe answer…"):
            try:
                # Pass history excluding the last user turn (it's in the RAG prompt)
                prior_history = st.session_state["chat_history"][:-1]
                answer = llm.generate_answer(
                    user_query=user_query,
                    retrieved_chunks=retrieved,
                    preferences=preferences,
                    chat_history=prior_history,
                )
            except ValueError as ve:
                answer = f"⚠️ **Configuration error:** {ve}"
            except Exception as exc:
                answer = f"❌ **API error:** {exc}"

        st.markdown(answer)

    # Append assistant turn (with sources for re-render)
    st.session_state["chat_history"].append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
