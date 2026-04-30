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

TRITON_RUNTIME_USABLE = TRITON_AVAILABLE and (Path(sysconfig.get_paths()["include"]) / "Python.h").exists()


def _round_up(value: int, multiple: int) -> int:
    if multiple <= 1:
        return value
    return int(math.ceil(value / multiple) * multiple)


def pack_left_padded_sequences_torch(
    sequences: list[torch.Tensor],
    *,
    pad_token_id: int,
    device: torch.device,
    pad_to_multiple_of: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not sequences:
        raise ValueError("pack_left_padded_sequences_torch requires at least one sequence")
    max_len = _round_up(max(int(sequence.numel()) for sequence in sequences), pad_to_multiple_of)
    output_ids = torch.full(
        (len(sequences), max_len),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    output_mask = torch.zeros(
        (len(sequences), max_len),
        dtype=torch.long,
        device=device,
    )
    for row_index, sequence in enumerate(sequences):
        length = int(sequence.numel())
        if length == 0:
            continue
        start = max_len - length
        row = sequence.to(device=device, dtype=torch.long)
        output_ids[row_index, start:] = row
        output_mask[row_index, start:] = 1
    return output_ids, output_mask


if TRITON_AVAILABLE:

    @triton.jit
    def _pack_left_padded_kernel(
        flat_ids_ptr,
        offsets_ptr,
        lengths_ptr,
        output_ids_ptr,
        output_mask_ptr,
        stride_ids_row,
        stride_mask_row,
        max_len,
        pad_token_id,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        block = tl.program_id(1)
        cols = block * BLOCK + tl.arange(0, BLOCK)
        valid_cols = cols < max_len

        seq_start = tl.load(offsets_ptr + row)
        seq_length = tl.load(lengths_ptr + row)
        left_pad = max_len - seq_length
        copy_mask = valid_cols & (cols >= left_pad)
        src_idx = seq_start + (cols - left_pad)

        values = tl.full((BLOCK,), pad_token_id, tl.int64)
        loaded = tl.load(flat_ids_ptr + src_idx, mask=copy_mask, other=pad_token_id)
        values = tl.where(copy_mask, loaded, values)
        tl.store(output_ids_ptr + row * stride_ids_row + cols, values, mask=valid_cols)

        mask_values = tl.where(copy_mask, 1, 0)
        tl.store(output_mask_ptr + row * stride_mask_row + cols, mask_values, mask=valid_cols)


def pack_left_padded_sequences(
    sequences: list[torch.Tensor],
    *,
    pad_token_id: int,
    device: torch.device,
    pad_to_multiple_of: int = 1,
    use_triton: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not sequences:
        raise ValueError("pack_left_padded_sequences requires at least one sequence")
    global TRITON_RUNTIME_USABLE
    if not use_triton or not TRITON_RUNTIME_USABLE or device.type != "cuda":
        return pack_left_padded_sequences_torch(
            sequences,
            pad_token_id=pad_token_id,
            device=device,
            pad_to_multiple_of=pad_to_multiple_of,
        )

    max_len = _round_up(max(int(sequence.numel()) for sequence in sequences), pad_to_multiple_of)
    flat_ids = torch.cat([sequence.to(device=device, dtype=torch.long) for sequence in sequences], dim=0)
    lengths = torch.tensor([int(sequence.numel()) for sequence in sequences], device=device, dtype=torch.int32)
    offsets = torch.zeros(len(sequences), device=device, dtype=torch.int32)
    if len(sequences) > 1:
        offsets[1:] = torch.cumsum(lengths[:-1], dim=0)

    output_ids = torch.empty((len(sequences), max_len), device=device, dtype=torch.long)
    output_mask = torch.empty((len(sequences), max_len), device=device, dtype=torch.long)
    block = 128
    grid = (len(sequences), triton.cdiv(max_len, block))
    try:
        _pack_left_padded_kernel[grid](
            flat_ids,
            offsets,
            lengths,
            output_ids,
            output_mask,
            output_ids.stride(0),
            output_mask.stride(0),
            max_len,
            pad_token_id,
            BLOCK=block,
        )
    except Exception:
        TRITON_RUNTIME_USABLE = False
        return pack_left_padded_sequences_torch(
            sequences,
            pad_token_id=pad_token_id,
            device=device,
            pad_to_multiple_of=pad_to_multiple_of,
        )
    return output_ids, output_mask
