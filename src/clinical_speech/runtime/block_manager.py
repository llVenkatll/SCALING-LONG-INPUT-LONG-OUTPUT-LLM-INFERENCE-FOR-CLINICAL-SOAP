from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass


class CacheBudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class KVCacheLayout:
    num_layers: int
    num_key_value_heads: int
    head_dim: int
    bytes_per_element: int

    @property
    def bytes_per_token(self) -> int:
        return (
            2
            * self.num_layers
            * self.num_key_value_heads
            * self.head_dim
            * self.bytes_per_element
        )


@dataclass
class BlockAllocation:
    request_id: str
    block_ids: list[int]
    tokens_used: int
    block_size_tokens: int

    @property
    def reserved_tokens(self) -> int:
        return len(self.block_ids) * self.block_size_tokens

    @property
    def slack_tokens(self) -> int:
        return self.reserved_tokens - self.tokens_used


class KVCacheBlockManager:
    def __init__(
        self,
        *,
        layout: KVCacheLayout,
        block_size_tokens: int,
        max_cache_budget_bytes: int,
    ):
        self.layout = layout
        self.block_size_tokens = block_size_tokens
        self.block_bytes = layout.bytes_per_token * block_size_tokens
        self.total_blocks = max(1, max_cache_budget_bytes // self.block_bytes)
        self.free_block_ids: deque[int] = deque(range(self.total_blocks))
        self.allocations: dict[str, BlockAllocation] = {}

    @property
    def total_capacity_tokens(self) -> int:
        return self.total_blocks * self.block_size_tokens

    def has_request(self, request_id: str) -> bool:
        return request_id in self.allocations

    def get_allocation(self, request_id: str) -> BlockAllocation:
        if request_id not in self.allocations:
            raise KeyError(f"Unknown KV allocation for request {request_id}")
        return self.allocations[request_id]

    def token_count(self, request_id: str) -> int:
        return self.get_allocation(request_id).tokens_used

    def block_table(self, request_id: str) -> list[int]:
        return list(self.get_allocation(request_id).block_ids)

    def _reserve_blocks(self, blocks_needed: int, *, request_id: str, action: str) -> list[int]:
        if blocks_needed > len(self.free_block_ids):
            raise CacheBudgetExceeded(
                f"KV cache budget exceeded while {action} request {request_id}: "
                f"need {blocks_needed} blocks but only {len(self.free_block_ids)} remain"
            )
        return [self.free_block_ids.popleft() for _ in range(blocks_needed)]

    def allocate(self, request_id: str, tokens: int) -> BlockAllocation:
        if request_id in self.allocations:
            raise ValueError(f"Request {request_id} already has a KV allocation")
        blocks_needed = max(1, math.ceil(tokens / self.block_size_tokens))
        block_ids = self._reserve_blocks(blocks_needed, request_id=request_id, action="allocating")
        allocation = BlockAllocation(
            request_id=request_id,
            block_ids=block_ids,
            tokens_used=tokens,
            block_size_tokens=self.block_size_tokens,
        )
        self.allocations[request_id] = allocation
        return allocation

    def ensure(self, request_id: str, tokens: int) -> BlockAllocation:
        if request_id not in self.allocations:
            return self.allocate(request_id, tokens)
        return self.grow(request_id, tokens)

    def grow(self, request_id: str, tokens: int) -> BlockAllocation:
        allocation = self.get_allocation(request_id)
        required_blocks = max(1, math.ceil(tokens / self.block_size_tokens))
        additional = required_blocks - len(allocation.block_ids)
        if additional > 0:
            allocation.block_ids.extend(
                self._reserve_blocks(additional, request_id=request_id, action="growing")
            )
        allocation.tokens_used = tokens
        return allocation

    def free(self, request_id: str) -> None:
        allocation = self.allocations.pop(request_id, None)
        if allocation is None:
            return
        for block_id in allocation.block_ids:
            self.free_block_ids.append(block_id)

    def snapshot(self) -> dict[str, float | int]:
        used_blocks = self.total_blocks - len(self.free_block_ids)
        allocated_bytes = used_blocks * self.block_bytes
        utilized_bytes = sum(
            allocation.tokens_used * self.layout.bytes_per_token
            for allocation in self.allocations.values()
        )
        slack_tokens = sum(allocation.slack_tokens for allocation in self.allocations.values())
        fragmentation_ratio = 0.0 if allocated_bytes == 0 else max(0.0, 1.0 - (utilized_bytes / allocated_bytes))
        return {
            "block_size_tokens": self.block_size_tokens,
            "total_blocks": self.total_blocks,
            "used_blocks": used_blocks,
            "free_blocks": len(self.free_block_ids),
            "allocated_bytes": allocated_bytes,
            "utilized_bytes": utilized_bytes,
            "utilization_ratio": 0.0 if allocated_bytes == 0 else utilized_bytes / allocated_bytes,
            "fragmentation_ratio": fragmentation_ratio,
            "slack_tokens": slack_tokens,
            "active_requests": len(self.allocations),
            "capacity_tokens": self.total_capacity_tokens,
        }
