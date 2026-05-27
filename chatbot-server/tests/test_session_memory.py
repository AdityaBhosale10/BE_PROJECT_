from src.adapters.memory.redis_memory import ChatTurn
from src.adapters.memory.session_memory import SessionMemoryStore
from src.services.conversation_context import build_conversation_context, summarize_assistant_turn


def test_session_memory_local_fallback_when_redis_unavailable():
    store = SessionMemoryStore(redis_url="redis://127.0.0.1:59999/0")
    assert store.backend == "memory"

    store.append_turn("s1", ChatTurn(role="user", content="running shoes"))
    store.append_turn("s1", ChatTurn(role="assistant", content='{"answer":"Try Brooks Ghost"}'))

    history = store.get_history("s1", limit=10)
    assert len(history) == 2
    assert history[0].content == "running shoes"


def test_build_conversation_context_summarizes_assistant_json():
    history = [
        ChatTurn(role="user", content="best running shoes"),
        ChatTurn(role="assistant", content='{"answer":"Brooks Ghost 15","products":[{"title":"Brooks Ghost"}]}'),
        ChatTurn(role="user", content="cheaper option?"),
    ]
    ctx = build_conversation_context(history, exclude_last_user=True)
    assert "best running shoes" in ctx
    assert "Brooks" in ctx
    assert "cheaper option" not in ctx


def test_summarize_assistant_turn_plain_text():
    assert summarize_assistant_turn("plain text answer") == "plain text answer"
