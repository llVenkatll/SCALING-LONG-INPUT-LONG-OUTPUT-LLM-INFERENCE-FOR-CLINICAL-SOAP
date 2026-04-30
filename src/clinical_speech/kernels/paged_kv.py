from __future__ import annotations

import math
import sysconfig
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    TRITON_AVAILABLE = False

TRITON_PAGED_KV_RUNTIME_USABLE = False
if TRITON_AVAILABLE:
    include_dir = Path(sysconfig.get_paths()["include"])
    TRITON_PAGED_KV_RUNTIME_USABLE = include_dir.joinpath("Python.h").exists()


def _paged_kv_launch_shape(
    *,
    batch_size: int,
    num_heads: int,
    output_length: int,
    head_dim: int,
    block_tokens: int,
    block_dmodel: int,
) -> tuple[tuple[int, int, int], int]:
    num_token_tiles = max(1, math.ceil(output_length / block_tokens))
    num_dim_tiles = max(1, math.ceil(head_dim / block_dmodel))
    return (batch_size, num_heads, num_token_tiles * num_dim_tiles), num_dim_tiles


def _unpack_paged_kv_tile_index(combined_tile_idx: int, *, num_dim_tiles: int) -> tuple[int, int]:
    return combined_tile_idx // num_dim_tiles, combined_tile_idx % num_dim_tiles


def max_blocks_for_lengths(lengths: list[int], block_size_tokens: int) -> int:
    return max(1, max(math.ceil(length / block_size_tokens) for length in lengths)) if lengths else 1


def gather_paged_kv_torch(
    pages: torch.Tensor,
    *,
    block_tables: list[list[int]],
    sequence_lengths: list[int],
    block_size_tokens: int,
    output_length: int,
) -> torch.Tensor:
    if len(block_tables) != len(sequence_lengths):
        raise ValueError("block_tables and sequence_lengths must have the same length")
    if output_length <= 0:
        raise ValueError("output_length must be positive")

    batch_size = len(sequence_lengths)
    num_heads = int(pages.shape[1])
    head_dim = int(pages.shape[3])
    output = torch.zeros(
        (batch_size, num_heads, output_length, head_dim),
        dtype=pages.dtype,
        device=pages.device,
    )
    for row_index, (block_ids, seq_length) in enumerate(zip(block_tables, sequence_lengths, strict=True)):
        if seq_length <= 0:
            continue
        if seq_length > output_length:
            raise ValueError(
                f"sequence length {seq_length} exceeds output length {output_length} for row {row_index}"
            )
        dest_start = output_length - seq_length
        copied = 0
        for block_id in block_ids:
            if copied >= seq_length:
                break
            span = min(block_size_tokens, seq_length - copied)
            output[row_index, :, dest_start + copied : dest_start + copied + span, :] = pages[
                block_id, :, :span, :
            ]
            copied += span
    return output


if TRITON_AVAILABLE:

    @triton.jit
    def _gather_paged_kv_kernel(
        pages_ptr,
        block_tables_ptr,
        sequence_lengths_ptr,
        output_ptr,
        pages_stride_block,
        pages_stride_head,
        pages_stride_token,
        pages_stride_dim,
        block_tables_stride_row,
        output_stride_batch,
        output_stride_head,
        output_stride_token,
        output_stride_dim,
        output_length,
        block_size_tokens,
        head_dim,
        num_dim_tiles,
        BLOCK_TOKENS: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        combined_tile_idx = tl.program_id(2)
        token_block_idx = combined_tile_idx // num_dim_tiles
        dim_block_idx = combined_tile_idx % num_dim_tiles

        token_offsets = token_block_idx * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
        dim_offsets = dim_block_idx * BLOCK_DMODEL + tl.arange(0, BLOCK_DMODEL)

        seq_length = tl.load(sequence_lengths_ptr + batch_idx)
        valid_tokens = token_offsets < output_length
        start_offset = output_length - seq_length
        logical_positions = token_offsets - start_offset
        live_tokens = valid_tokens & (logical_positions >= 0) & (logical_positions < seq_length)

        block_indices = logical_positions // block_size_tokens
        slot_indices = logical_positions % block_size_tokens
        block_ids = tl.load(
            block_tables_ptr + batch_idx * block_tables_stride_row + block_indices,
            mask=live_tokens,
            other=0,
        )

        page_offsets = (
            block_ids[:, None] * pages_stride_block
            + head_idx * pages_stride_head
            + slot_indices[:, None] * pages_stride_token
            + dim_offsets[None, :] * pages_stride_dim
        )
        dim_mask = dim_offsets < head_dim
        values = tl.load(
            pages_ptr + page_offsets,
            mask=live_tokens[:, None] & dim_mask[None, :],
            other=0.0,
        )

        output_offsets = (
            batch_idx * output_stride_batch
            + head_idx * output_stride_head
            + token_offsets[:, None] * output_stride_token
            + dim_offsets[None, :] * output_stride_dim
        )
        tl.store(
            output_ptr + output_offsets,
            values,
            mask=valid_tokens[:, None] & dim_mask[None, :],
        )


def _materialize_block_tables(
    block_tables: list[list[int]],
    *,
    device: torch.device,
) -> torch.Tensor:
    max_blocks = max(len(block_ids) for block_ids in block_tables)
    table_tensor = torch.zeros((len(block_tables), max_blocks), dtype=torch.int32, device=device)
    for row_index, block_ids in enumerate(block_tables):
        if block_ids:
            table_tensor[row_index, : len(block_ids)] = torch.tensor(block_ids, dtype=torch.int32, device=device)
    return table_tensor


def gather_paged_kv_triton(
    pages: torch.Tensor,
    *,
    block_tables: list[list[int]],
    sequence_lengths: list[int],
    block_size_tokens: int,
    output_length: int,
) -> torch.Tensor:
    if not TRITON_PAGED_KV_RUNTIME_USABLE:
        raise RuntimeError("Triton paged KV gather is not runtime-usable on this host")
    if pages.device.type != "cuda":
        raise ValueError("gather_paged_kv_triton requires CUDA tensors")

    output = torch.zeros(
        (len(sequence_lengths), int(pages.shape[1]), output_length, int(pages.shape[3])),
        dtype=pages.dtype,
        device=pages.device,
    )
    block_tables_tensor = _materialize_block_tables(block_tables, device=pages.device)
    sequence_lengths_tensor = torch.tensor(sequence_lengths, dtype=torch.int32, device=pages.device)

    block_tokens = 32
    block_dmodel = 32
    grid, num_dim_tiles = _paged_kv_launch_shape(
        batch_size=len(sequence_lengths),
        num_heads=int(pages.shape[1]),
        output_length=output_length,
        head_dim=int(pages.shape[3]),
        block_tokens=block_tokens,
        block_dmodel=block_dmodel,
    )
    _gather_paged_kv_kernel[grid](
        pages,
        block_tables_tensor,
        sequence_lengths_tensor,
        output,
        pages.stride(0),
        pages.stride(1),
        pages.stride(2),
        pages.stride(3),
        block_tables_tensor.stride(0),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        output_length,
        block_size_tokens,
        int(pages.shape[3]),
        num_dim_tiles,
        BLOCK_TOKENS=block_tokens,
        BLOCK_DMODEL=block_dmodel,
    )
    return output


def gather_paged_kv(
    pages: torch.Tensor,
    *,
    block_tables: list[list[int]],
    sequence_lengths: list[int],
    block_size_tokens: int,
    output_length: int,
    use_triton: bool = False,
) -> torch.Tensor:
    if use_triton and TRITON_PAGED_KV_RUNTIME_USABLE and pages.device.type == "cuda":
        return gather_paged_kv_triton(
            pages,
            block_tables=block_tables,
            sequence_lengths=sequence_lengths,
            block_size_tokens=block_size_tokens,
            output_length=output_length,
        )
    return gather_paged_kv_torch(
        pages,
        block_tables=block_tables,
        sequence_lengths=sequence_lengths,
        block_size_tokens=block_size_tokens,
        output_length=output_length,
    )
