from __future__ import annotations

import json
import re
from typing import List, Optional

from src.adapters.memory.redis_memory import ChatTurn
from typing import Tuple


def summarize_history(history: List[ChatTurn], llm_client, llm_model: str, keep_recent: int = 6) -> Tuple[List[ChatTurn], bool]:
    """
    Summarize older turns in history into a single compressed assistant turn.

    - Keeps the latest `keep_recent` turns untouched.
    - Compresses older turns into one assistant `ChatTurn` with role 'assistant'
      and content equal to a short JSON summary string.

    Returns (new_history, compressed_flag).
    """
    if not history or len(history) <= keep_recent:
        return history, False

    # split older and recent
    older = history[:-keep_recent]
    recent = history[-keep_recent:]

    # build a prompt summarizing the older conversation
    lines = []
    for turn in older:
        if turn.role == "user":
            lines.append(f"User: {turn.content}")
        else:
            # assistant turns may be JSON; use summarize_assistant_turn
            lines.append(f"Assistant: {summarize_assistant_turn(turn.content)}")

    prompt = (
        "You are an assistant that summarizes past conversation context for later use. "
        "Produce a concise JSON summary (fields: summary, important_entities, recent_recommendations) representing the conversation below. "
        "Keep it short (<= 300 chars) and preserve product names, filters and decisions.\n\n"
        "Conversation:\n"
        + "\n".join(lines)
        + "\n\nReturn only a JSON object."
    )

    try:
        summary_text = llm_client.generate(prompt=prompt, model=llm_model)
        summary_json = summarize_assistant_turn(summary_text)
    except Exception:
        # fallback: join plain lines
        summary_json = " ".join(lines)[:300]

    compressed_turn = ChatTurn(role="assistant", content=json.dumps({"summary": summary_json}))
    new_history = [compressed_turn] + recent
    return new_history, True


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def summarize_assistant_turn(content: str) -> str:
    """Turn stored assistant JSON into a short line for follow-up prompts."""
    try:
        data = json.loads(_strip_code_fences(content))
        parts = []
        if isinstance(data, dict):
            if data.get("answer"):
                parts.append(str(data["answer"]))
            if data.get("initial", {}).get("message"):
                parts.append(str(data["initial"]["message"]))
            if data.get("final", {}).get("message"):
                parts.append(str(data["final"]["message"]))
            products = data.get("products") or []
            if products:
                titles = [p.get("title") or p.get("name") for p in products[:3] if isinstance(p, dict)]
                titles = [t for t in titles if t]
                if titles:
                    parts.append("Recommended: " + ", ".join(titles))
        if parts:
            return " ".join(parts)[:500]
    except Exception:
        pass
    return content[:500]


def build_conversation_context(history: List[ChatTurn], exclude_last_user: bool = True) -> str:
    """
    Format prior turns for the LLM (excludes the latest user message when exclude_last_user=True).
    """
    if not history:
        return ""

    turns = history[:-1] if exclude_last_user and history and history[-1].role == "user" else history
    lines: List[str] = []
    for turn in turns:
        if turn.role == "user":
            lines.append(f"User: {turn.content}")
        elif turn.role == "assistant":
            lines.append(f"Assistant: {summarize_assistant_turn(turn.content)}")
    return "\n".join(lines).strip()


def build_effective_user_query(user_query: str, conversation_context: str) -> str:
    if not conversation_context:
        return user_query
    return (
        "You are continuing an ongoing product-research conversation.\n"
        "Use the conversation context to resolve references like 'it', 'those', 'cheaper', 'the previous one'.\n\n"
        f"Conversation context:\n{conversation_context}\n\n"
        f"Current user message:\n{user_query}"
    )
