import time
from collections import deque


class RateLimiter:
    """Simple sliding-window rate limiter — N events per 60 s, per key."""

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self._buckets: dict[str, deque[float]] = {}

    def allow(self, key: str) -> bool:
        if self.per_minute <= 0:
            return True
        now = time.time()
        cutoff = now - 60.0
        bucket = self._buckets.setdefault(key, deque())
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= self.per_minute:
            return False
        bucket.append(now)
        return True

    def reset_in(self, key: str) -> float:
        bucket = self._buckets.get(key)
        if not bucket:
            return 0.0
        return max(0.0, 60.0 - (time.time() - bucket[0]))
