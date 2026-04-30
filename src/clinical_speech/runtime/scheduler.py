from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class ScheduledBatch:
    batch_index: int
    request_ids: list[str]
    prompts: list[str]


@dataclass(frozen=True)
class PendingRequest:
    request_id: str
    prompt: str
    enqueued_at: float = 0.0


class StaticBatchScheduler:
    def __init__(self, max_batch_size: int):
        self.max_batch_size = max_batch_size

    def schedule(self, prompts: Iterable[str]) -> list[ScheduledBatch]:
        items = list(prompts)
        batches: list[ScheduledBatch] = []
        for batch_index, start in enumerate(range(0, len(items), self.max_batch_size)):
            chunk = items[start : start + self.max_batch_size]
            batches.append(
                ScheduledBatch(
                    batch_index=batch_index,
                    request_ids=[f"req_{start + offset:05d}" for offset in range(len(chunk))],
                    prompts=chunk,
                )
            )
        return batches


class RequestQueue:
    def __init__(self):
        self._items: deque[PendingRequest] = deque()

    def __len__(self) -> int:
        return len(self._items)

    def push(self, request: PendingRequest) -> None:
        self._items.append(request)

    def pop_many(self, limit: int) -> list[PendingRequest]:
        items: list[PendingRequest] = []
        while self._items and len(items) < limit:
            items.append(self._items.popleft())
        return items


class QueueAdmissionScheduler:
    def __init__(self, *, max_batch_size: int, max_concurrent_requests: int):
        self.max_batch_size = max_batch_size
        self.max_concurrent_requests = max_concurrent_requests

    def admit(self, queue: RequestQueue, *, active_requests: int) -> list[PendingRequest]:
        available = min(self.max_batch_size, max(0, self.max_concurrent_requests - active_requests))
        if available <= 0:
            return []
        return queue.pop_many(available)
