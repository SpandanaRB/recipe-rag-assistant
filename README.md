# 🍳 Recipe RAG Assistant

An intelligent recipe generator built with **Retrieval-Augmented Generation (RAG)**.  
Ask natural-language cooking questions and get personalized, context-grounded answers — complete with step-by-step instructions, substitutions, nutrition facts, and a shopping list.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend / App | [Streamlit](https://streamlit.io) |
| LLM | [Groq API](https://console.groq.com) · model `openai/gpt-oss-120b` |
| Embeddings | `sentence-transformers` · `all-MiniLM-L6-v2` |
| Vector Store | `faiss-cpu` (in-memory, persisted in `st.session_state`) |
| PDF parsing | `pypdf` |
| Config | `python-dotenv` |

---

## Project Structure

```
.
├── app.py               # Main Streamlit app
├── rag_utils.py         # Chunking, embedding, FAISS index, retrieval
├── llm_utils.py         # Groq API wrapper & prompt templates
├── sample_data/         # 20 pre-built recipe .txt files
│   ├── chocolate_cake.txt
│   ├── chicken_tikka_masala.txt
│   └── ...
├── requirements.txt
├── .env.example         # Template for your API key
└── README.md
```

---

## Setup

### 1. Prerequisites

- Python 3.11+
- A free [Groq API key](https://console.groq.com/keys)

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `faiss-cpu` and `sentence-transformers` will each download model weights on first run (~90 MB total). An internet connection is required the first time.

### 3. Set your Groq API key

Copy `.env.example` to `.env` and fill in your key:

```bash
cp .env.example .env
```

Edit `.env`:

```
GROQ_API_KEY=your_groq_api_key_here
```

Alternatively, export it as an environment variable:

```bash
export GROQ_API_KEY=your_groq_api_key_here   # macOS / Linux
$env:GROQ_API_KEY="your_groq_api_key_here"   # Windows PowerShell
```

### 4. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Features

### 📂 Document Ingestion
- **Sample recipes** — click *Load Sample Recipes* in the sidebar to instantly index 20 pre-built recipes covering Italian, Indian, Mexican, Thai, Greek, Japanese, French, and more.
- **Custom upload** — drag-and-drop any `.txt` or `.pdf` recipe files. They are chunked, embedded, and added to the in-memory FAISS index automatically.

### 💬 Conversational Q&A (RAG)
- Chat interface powered by `st.chat_input` / `st.chat_message`.
- Top-K relevant chunks are retrieved from FAISS and injected into the LLM prompt.
- Source recipe names are displayed above every answer for transparency.
- Conversation history (last 6 turns) is retained for follow-up questions.

### 🎛️ Personalization
Configure in the sidebar:
- **Dietary restrictions** — Vegan, Gluten-Free, Diabetic-Friendly, Keto, etc.
- **Available ingredients** — the LLM skips these in the shopping list.
- **Max cooking time** — answers are filtered/adapted accordingly.
- **Cuisine preference** — biases recipe suggestions.

### 📋 Structured Output
Every answer is formatted as:
1. **Recipe Answer** — personalised intro
2. **Step-by-Step Instructions** — numbered
3. **Substitutions** — table of swaps for allergens or missing items
4. **Nutrition Facts** — per-serving table (estimated)
5. **Shopping List** — items not already in your available ingredients

---

## Example Questions

- *"How do I make a vegan chocolate cake?"*
- *"Give me a gluten-free dinner I can make in under 30 minutes."*
- *"What can I cook with eggs, spinach, and feta?"*
- *"Suggest a diabetic-friendly breakfast."*
- *"How can I adapt chicken tikka masala for a dairy-free diet?"*

---

## Configuration Reference

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Your Groq API key (required) |

All other settings (chunk size, overlap, embedding model, top-K) can be tuned in `rag_utils.py`.

---

## License

MIT
