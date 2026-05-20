from src.adapters.memory.redis_memory import ChatTurn, RedisMemory


class FakeRedis:
    def __init__(self):
        self.store = {}

    def rpush(self, key, value):
        self.store.setdefault(key, []).append(value)

    def ltrim(self, key, start, end):
        arr = self.store.get(key, [])
        self.store[key] = arr[start if start >= 0 else max(len(arr) + start, 0) : end + 1 if end != -1 else None]

    def lrange(self, key, start, end):
        arr = self.store.get(key, [])
        return arr[start if start >= 0 else max(len(arr) + start, 0) : end + 1 if end != -1 else None]


def test_redis_memory_append_and_get_history() -> None:
    fake = FakeRedis()
    mem = RedisMemory(client=fake, max_turns=3)

    mem.append_turn("s1", ChatTurn(role="user", content="hi"))
    mem.append_turn("s1", ChatTurn(role="assistant", content="hello"))
    mem.append_turn("s1", ChatTurn(role="user", content="q"))
    mem.append_turn("s1", ChatTurn(role="assistant", content="a"))

    hist = mem.get_history("s1", limit=10)
    assert [t.role for t in hist] == ["assistant", "user", "assistant"]
