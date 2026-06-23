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

    def edit_turn(self, session_id: str, index: int, new_turn: ChatTurn) -> bool:
        """
        Edit a specific turn by index (0-based from oldest). Returns True if edited.
        If index is negative it counts from the end like Python indexing.
        """
        if not session_id:
            return False

        # Work with Redis if available
        if self._redis is not None:
            try:
                turns = self._redis.get_history(session_id, limit=self.max_turns)
                if not turns:
                    return False
                # normalize negative index
                if index < 0:
                    index = len(turns) + index
                if index < 0 or index >= len(turns):
                    return False
                turns[index] = new_turn
                self._redis.set_history(session_id, turns)
                return True
            except Exception as exc:
                logger.warning("Redis edit failed, falling back to in-memory: %s", exc)

        # Fallback to local store
        bucket = self._local.get(session_id, [])
        if not bucket:
            return False
        if index < 0:
            index = len(bucket) + index
        if index < 0 or index >= len(bucket):
            return False
        bucket[index] = new_turn
        # trim if necessary
        if len(bucket) > self.max_turns:
            bucket = bucket[-self.max_turns :]
            self._local[session_id] = bucket
        return True

    def delete_turn(self, session_id: str, index: int) -> bool:
        """
        Delete a specific turn by index (0-based from oldest). Returns True if deleted.
        """
        if not session_id:
            return False

        if self._redis is not None:
            try:
                turns = self._redis.get_history(session_id, limit=self.max_turns)
                if not turns:
                    return False
                if index < 0:
                    index = len(turns) + index
                if index < 0 or index >= len(turns):
                    return False
                del turns[index]
                self._redis.set_history(session_id, turns)
                return True
            except Exception as exc:
                logger.warning("Redis delete failed, falling back to in-memory: %s", exc)

        bucket = self._local.get(session_id, [])
        if not bucket:
            return False
        if index < 0:
            index = len(bucket) + index
        if index < 0 or index >= len(bucket):
            return False
        del bucket[index]
        self._local[session_id] = bucket
        return True

    def set_history(self, session_id: str, turns: List[ChatTurn]) -> None:
        """
        Replace the full session history with the provided turns.
        """
        if not session_id:
            return
        if self._redis is not None:
            try:
                self._redis.set_history(session_id, turns)
                return
            except Exception as exc:
                logger.warning("Redis set_history failed, falling back to local: %s", exc)

        # Store locally
        self._local[session_id] = turns[-self.max_turns :]

    # Reactions support (Redis hash or in-process dict)
    def set_reaction(self, session_id: str, index: int, reaction: Optional[str]) -> None:
        if not session_id:
            return
        # Redis-backed
        if self._redis is not None:
            try:
                key = f"chat:{session_id}:reactions"
                # store as string; delete if None
                if reaction is None:
                    self._redis._client.hdel(key, str(index))
                else:
                    self._redis._client.hset(key, str(index), reaction)
                return
            except Exception as exc:
                logger.warning("Redis set_reaction failed, falling back to local: %s", exc)

        # in-process
        meta = self._local.setdefault(f"__reactions__{session_id}", {})
        if reaction is None:
            meta.pop(str(index), None)
        else:
            meta[str(index)] = reaction

    def get_reactions(self, session_id: str) -> Dict[str, str]:
        if not session_id:
            return {}
        if self._redis is not None:
            try:
                key = f"chat:{session_id}:reactions"
                data = self._redis._client.hgetall(key) or {}
                return {k: v for k, v in data.items()}
            except Exception as exc:
                logger.warning("Redis get_reactions failed, falling back to local: %s", exc)

        return dict(self._local.get(f"__reactions__{session_id}", {}))
