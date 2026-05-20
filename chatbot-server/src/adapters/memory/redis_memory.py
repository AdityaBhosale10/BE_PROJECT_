from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ChatTurn:
    role: str  # "user" | "assistant"
    content: str


class RedisMemory:
    """
    Redis-backed chat history store.

    Stores turns in a Redis list: `chat:{session_id}`.
    """

    def __init__(self, redis_url: Optional[str] = None, max_turns: int = 50, client: Any = None) -> None:
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.max_turns = max_turns
        self._client = client

        if self._client is None:
            import redis  # type: ignore

            self._client = redis.Redis.from_url(self.redis_url, decode_responses=True)

    def append_turn(self, session_id: str, turn: ChatTurn) -> None:
        key = self._key(session_id)
        self._client.rpush(key, json.dumps({"role": turn.role, "content": turn.content}))
        # keep last N
        self._client.ltrim(key, -self.max_turns, -1)

    def get_history(self, session_id: str, limit: int = 20) -> List[ChatTurn]:
        key = self._key(session_id)
        raw = self._client.lrange(key, -limit, -1) or []
        out: List[ChatTurn] = []
        for item in raw:
            try:
                obj = json.loads(item)
                out.append(ChatTurn(role=obj.get("role", "user"), content=obj.get("content", "")))
            except Exception:
                continue
        return out

    @staticmethod
    def _key(session_id: str) -> str:
        return f"chat:{session_id}"

