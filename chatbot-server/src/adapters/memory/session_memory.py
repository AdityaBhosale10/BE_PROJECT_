from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional

from src.adapters.memory.redis_memory import ChatTurn, RedisMemory

logger = logging.getLogger(__name__)


class SessionMemoryStore:
    """
    Session memory with Redis primary storage and in-process fallback.

    Chat keeps working when Redis is not installed or not running.
    """

    def __init__(self, redis_url: Optional[str] = None, max_turns: int = 50) -> None:
        self.max_turns = max_turns
        self._redis: Optional[RedisMemory] = None
        self._local: Dict[str, List[ChatTurn]] = {}

        if redis_url:
            try:
                import redis

                client = redis.Redis.from_url(redis_url, decode_responses=True, socket_connect_timeout=2)
                client.ping()
                self._redis = RedisMemory(redis_url=redis_url, max_turns=max_turns, client=client)
                logger.info("Session memory: using Redis at %s", redis_url)
            except Exception as exc:
                logger.warning("Session memory: Redis unavailable (%s). Using in-process fallback.", exc)

        if self._redis is None:
            logger.info("Session memory: using in-process store")

    @property
    def backend(self) -> str:
        return "redis" if self._redis is not None else "memory"

    def append_turn(self, session_id: str, turn: ChatTurn) -> None:
        if not session_id:
            return
        if self._redis is not None:
            try:
                self._redis.append_turn(session_id, turn)
                return
            except Exception as exc:
                logger.warning("Redis append failed, using local fallback: %s", exc)

        bucket = self._local.setdefault(session_id, [])
        bucket.append(turn)
        if len(bucket) > self.max_turns:
            self._local[session_id] = bucket[-self.max_turns :]

    def get_history(self, session_id: str, limit: int = 20) -> List[ChatTurn]:
        if not session_id:
            return []
        if self._redis is not None:
            try:
                return self._redis.get_history(session_id, limit=limit)
            except Exception as exc:
                logger.warning("Redis read failed, using local fallback: %s", exc)

        return list(self._local.get(session_id, [])[-limit:])
