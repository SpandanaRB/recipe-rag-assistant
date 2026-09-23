"""
llm_utils.py
Groq API wrapper and prompt-template helpers for Recipe RAG Assistant.
"""

from __future__ import annotations

import os
from typing import List

from groq import Groq

MODEL = "openai/gpt-oss-120b"

# ── Client factory ────────────────────────────────────────────────────────────

def get_groq_client() -> Groq:
    """Return a Groq client using GROQ_API_KEY from environment."""
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. Add it to your .env file or export it as an "
            "environment variable before running the app."
        )
    return Groq(api_key=api_key)


# ── Prompt builders ───────────────────────────────────────────────────────────

def build_system_prompt(preferences: dict) -> str:
    """
    Build the system prompt that encodes the user's dietary preferences.
    `preferences` keys: dietary_restrictions, available_ingredients,
                        max_cook_time, cuisine_preference.
    """
    dietary = preferences.get("dietary_restrictions") or "None specified"
    ingredients = preferences.get("available_ingredients") or "Not specified"
    max_time = preferences.get("max_cook_time") or "No limit"
    cuisine = preferences.get("cuisine_preference") or "Any"

    return f"""You are an expert culinary assistant and recipe generator. \
Your job is to answer recipe-related questions using the provided recipe context.

## User Preferences (ALWAYS respect these):
- Dietary restrictions / health requirements: {dietary}
- Ingredients already available: {ingredients}
- Maximum total cooking time: {max_time} minutes
- Preferred cuisine style: {cuisine}

## Response Format (ALWAYS use this exact structure):

### 🍽️ Recipe Answer
Provide a clear, friendly answer to the user's question, adapted to their \
preferences. If the original recipe conflicts with dietary restrictions, \
suggest a fully compliant alternative or modification.

### 📋 Step-by-Step Instructions
Numbered steps that are clear and easy to follow.

### 🔄 Substitutions
A concise box (use a markdown table or bullet list) of possible ingredient \
substitutions — especially for allergens, dietary needs, or unavailable items.

### 🥗 Estimated Nutrition Facts (per serving)
Present as a simple markdown table with: Calories, Protein, Carbohydrates, \
Fat, Fiber, Sugar.

### 🛒 Shopping List
Bullet list of ingredients the user still needs to buy (i.e., NOT in their \
available ingredients list). If you don't know what's available, list all \
main ingredients.

Keep the tone friendly, practical, and encouraging. If context is insufficient, \
say so honestly rather than inventing a recipe."""


def build_rag_prompt(user_query: str, retrieved_chunks: List[dict]) -> str:
    """
    Build the user-turn message that includes retrieved context + the query.
    """
    if not retrieved_chunks:
        context_block = "No recipe documents are loaded yet."
    else:
        parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            source = chunk.get("source", "Unknown")
            text = chunk.get("text", "").strip()
            parts.append(f"[Context {i} — Source: {source}]\n{text}")
        context_block = "\n\n---\n\n".join(parts)

    return f"""## Retrieved Recipe Context:

{context_block}

---

## User Question:
{user_query}

Please answer using the recipe context above. Adapt the response to my dietary \
preferences and available ingredients as instructed in the system prompt."""


# ── LLM call ──────────────────────────────────────────────────────────────────

def generate_answer(
    user_query: str,
    retrieved_chunks: List[dict],
    preferences: dict,
    chat_history: List[dict] | None = None,
) -> str:
    """
    Call the Groq API and return the assistant's answer as a string.

    Parameters
    ----------
    user_query       : The user's latest question.
    retrieved_chunks : Top-k chunks from FAISS retrieval.
    preferences      : Dict of sidebar personalization settings.
    chat_history     : Optional prior turns as [{"role": ..., "content": ...}].
    """
    client = get_groq_client()

    system_prompt = build_system_prompt(preferences)
    user_message = build_rag_prompt(user_query, retrieved_chunks)

    messages: List[dict] = [{"role": "system", "content": system_prompt}]

    # Inject limited prior turns for conversational continuity (last 6 turns)
    if chat_history:
        for turn in chat_history[-6:]:
            role = turn.get("role")
            content = turn.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=2048,
    )

    return response.choices[0].message.content or ""


# ── Utility ───────────────────────────────────────────────────────────────────

def check_api_key() -> bool:
    """Return True if GROQ_API_KEY is present in environment."""
    return bool(os.getenv("GROQ_API_KEY", "").strip())
