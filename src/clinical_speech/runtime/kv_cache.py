from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers.cache_utils import Cache, CacheLayerMixin

from clinical_speech.kernels.paged_kv import gather_paged_kv
from clinical_speech.runtime.block_manager import KVCacheBlockManager, KVCacheLayout


def append_attention_tokens(attention_mask: torch.Tensor, num_new_tokens: int = 1) -> torch.Tensor:
    pad = torch.ones(
        (attention_mask.shape[0], num_new_tokens),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    return torch.cat([attention_mask, pad], dim=1)


def position_ids_from_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    return position_ids.masked_fill(attention_mask == 0, 0)


def infer_mistral_kv_layout(model, dtype: torch.dtype | None = None) -> KVCacheLayout:
    config = model.config
    dtype = dtype or getattr(model, "dtype", torch.float16)
    head_dim = int(config.hidden_size // config.num_attention_heads)
    return KVCacheLayout(
        num_layers=int(config.num_hidden_layers),
        num_key_value_heads=int(config.num_key_value_heads),
        head_dim=head_dim,
        bytes_per_element=torch.empty((), dtype=dtype).element_size(),
    )


@dataclass(frozen=True)
class CacheBatchView:
    request_ids: list[str]
    block_tables: list[list[int]]
    base_lengths: list[int]
    pending_lengths: list[int]
    planned_lengths: list[int]
    attention_mask_length: int

    @property
    def max_base_length(self) -> int:
        return max(self.base_lengths, default=0)

    @property
    def max_planned_length(self) -> int:
        return max(self.planned_lengths, default=0)


class PagedKVCacheLayer(CacheLayerMixin):
    is_compileable = False
    is_sliding = False

    def __init__(
        self,
        *,
        block_manager: KVCacheBlockManager,
        block_size_tokens: int,
        use_triton_gather: bool = False,
    ):
        super().__init__()
        self.block_manager = block_manager
        self.block_size_tokens = block_size_tokens
        self.use_triton_gather = use_triton_gather
        self.max_cache_len = block_manager.total_capacity_tokens
        self.max_batch_size = 0
        self.device = torch.device("cpu")
        self.dtype = torch.float16
        self.num_heads = 0
        self.head_dim = 0
        self._batch_view: CacheBatchView | None = None

    def set_batch_view(self, batch_view: CacheBatchView | None) -> None:
        self._batch_view = batch_view

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.dtype = key_states.dtype
        self.device = key_states.device
        self.num_heads = int(key_states.shape[1])
        self.head_dim = int(key_states.shape[-1])
        shape = (
            self.block_manager.total_blocks,
            self.num_heads,
            self.block_size_tokens,
            self.head_dim,
        )
        self.keys = torch.zeros(shape, dtype=self.dtype, device=self.device)
        self.values = torch.zeros(shape, dtype=value_states.dtype, device=self.device)
        self.is_initialized = True

    def _write_request_tokens(
        self,
        *,
        page_tensor: torch.Tensor,
        state_tensor: torch.Tensor,
        block_ids: list[int],
        base_length: int,
        pending_length: int,
    ) -> None:
        if pending_length <= 0:
            return
        src = state_tensor[:, state_tensor.shape[-2] - pending_length :, :]
        copied = 0
        logical_position = base_length
        while copied < pending_length:
            block_offset = logical_position // self.block_size_tokens
            slot_offset = logical_position % self.block_size_tokens
            span = min(self.block_size_tokens - slot_offset, pending_length - copied)
            block_id = block_ids[block_offset]
            page_tensor[block_id, :, slot_offset : slot_offset + span, :].copy_(
                src[:, copied : copied + span, :]
            )
            copied += span
            logical_position += span

    def _write_pending_tokens(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        if self._batch_view is None:
            raise RuntimeError("PagedKVCacheLayer requires an active batch view before update()")
        for row_index, (block_ids, base_length, pending_length) in enumerate(
            zip(
                self._batch_view.block_tables,
                self._batch_view.base_lengths,
                self._batch_view.pending_lengths,
                strict=True,
            )
        ):
            self._write_request_tokens(
                page_tensor=self.keys,
                state_tensor=key_states[row_index],
                block_ids=block_ids,
                base_length=base_length,
                pending_length=pending_length,
            )
            self._write_request_tokens(
                page_tensor=self.values,
                state_tensor=value_states[row_index],
                block_ids=block_ids,
                base_length=base_length,
                pending_length=pending_length,
            )

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del args, kwargs
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        if self._batch_view is None:
            raise RuntimeError("PagedKVCacheLayer.update called without batch metadata")

        self.max_batch_size = max(self.max_batch_size, int(key_states.shape[0]))
        self._write_pending_tokens(key_states, value_states)

        query_length = int(key_states.shape[-2])
        # For prompt prefill we can return the dense projected states directly and still populate the page pool.
        if self._batch_view.max_base_length == 0 and query_length == self._batch_view.attention_mask_length:
            return key_states, value_states
        if query_length != 1:
            raise NotImplementedError(
                "PagedKVCacheLayer currently supports prefill or single-token decode steps only"
            )

        dense_keys = gather_paged_kv(
            self.keys,
            block_tables=self._batch_view.block_tables,
            sequence_lengths=self._batch_view.planned_lengths,
            block_size_tokens=self.block_size_tokens,
            output_length=self._batch_view.attention_mask_length,
            use_triton=self.use_triton_gather,
        )
        dense_values = gather_paged_kv(
            self.values,
            block_tables=self._batch_view.block_tables,
            sequence_lengths=self._batch_view.planned_lengths,
            block_size_tokens=self.block_size_tokens,
            output_length=self._batch_view.attention_mask_length,
            use_triton=self.use_triton_gather,
        )
        return dense_keys, dense_values

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        if self._batch_view is None:
            return query_length, 0
        return self._batch_view.attention_mask_length, 0

    def get_seq_length(self) -> int:
        if self._batch_view is None:
            return 0
        return self._batch_view.max_base_length

    def get_max_cache_shape(self) -> int:
        return self.max_cache_len

    def crop(self, max_length: int) -> None:
        del max_length
        raise NotImplementedError("PagedKVCacheLayer does not support crop()")

    def batch_repeat_interleave(self, repeats: int) -> None:
        del repeats
        raise NotImplementedError("PagedKVCacheLayer does not support batch_repeat_interleave()")

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        del indices
        raise NotImplementedError("PagedKVCacheLayer does not support batch_select_indices() directly")


class PagedKVCache(Cache):
    def __init__(
        self,
        *,
        layout: KVCacheLayout,
        block_manager: KVCacheBlockManager,
        block_size_tokens: int,
        use_triton_gather: bool = False,
    ):
        self.layout = layout
        self.block_manager = block_manager
        self.block_size_tokens = block_size_tokens
        self.use_triton_gather = use_triton_gather
        layers = [
            PagedKVCacheLayer(
                block_manager=block_manager,
                block_size_tokens=block_size_tokens,
                use_triton_gather=use_triton_gather,
            )
            for _ in range(layout.num_layers)
        ]
        super().__init__(layers=layers)
        self._batch_view: CacheBatchView | None = None

    def begin_forward(
        self,
        *,
        request_ids: list[str],
        token_lengths: list[int],
        attention_mask_length: int,
    ) -> CacheBatchView:
        if len(request_ids) != len(token_lengths):
            raise ValueError("request_ids and token_lengths must have the same length")

        base_lengths: list[int] = []
        planned_lengths: list[int] = []
        block_tables: list[list[int]] = []
        for request_id, new_tokens in zip(request_ids, token_lengths, strict=True):
            base_length = self.block_manager.token_count(request_id) if self.block_manager.has_request(request_id) else 0
            planned_length = base_length + new_tokens
            self.block_manager.ensure(request_id, planned_length)
            base_lengths.append(base_length)
            planned_lengths.append(planned_length)
            block_tables.append(self.block_manager.block_table(request_id))

        batch_view = CacheBatchView(
            request_ids=list(request_ids),
            block_tables=block_tables,
            base_lengths=base_lengths,
            pending_lengths=list(token_lengths),
            planned_lengths=planned_lengths,
            attention_mask_length=attention_mask_length,
        )
        self._batch_view = batch_view
        for layer in self.layers:
            layer.set_batch_view(batch_view)
        return batch_view

    def clear_batch_view(self) -> None:
        self._batch_view = None
        for layer in self.layers:
            layer.set_batch_view(None)

    def free_requests(self, request_ids: list[str]) -> None:
        for request_id in request_ids:
            self.block_manager.free(request_id)

    def snapshot(self) -> dict[str, float | int]:
        return self.block_manager.snapshot()
