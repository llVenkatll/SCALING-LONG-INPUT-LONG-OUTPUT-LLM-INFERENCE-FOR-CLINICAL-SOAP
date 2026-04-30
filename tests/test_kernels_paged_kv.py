import unittest

import torch

from clinical_speech.kernels.paged_kv import (
    TRITON_PAGED_KV_RUNTIME_USABLE,
    _paged_kv_launch_shape,
    _unpack_paged_kv_tile_index,
    gather_paged_kv,
    gather_paged_kv_torch,
)


class PagedKVKernelTest(unittest.TestCase):
    def test_launch_shape_uses_only_three_triton_axes(self) -> None:
        grid, num_dim_tiles = _paged_kv_launch_shape(
            batch_size=2,
            num_heads=8,
            output_length=65,
            head_dim=96,
            block_tokens=32,
            block_dmodel=32,
        )
        self.assertEqual(len(grid), 3)
        self.assertEqual(grid[0], 2)
        self.assertEqual(grid[1], 8)
        self.assertEqual(num_dim_tiles, 3)
        self.assertEqual(grid[2], 9)

    def test_combined_tile_index_round_trips(self) -> None:
        num_dim_tiles = 4
        recovered = [_unpack_paged_kv_tile_index(idx, num_dim_tiles=num_dim_tiles) for idx in range(12)]
        expected = [(token_idx, dim_idx) for token_idx in range(3) for dim_idx in range(4)]
        self.assertEqual(recovered, expected)

    def test_gather_paged_kv_reference_right_aligns_sequences(self) -> None:
        pages = torch.tensor(
            [
                [[[1.0], [2.0]]],
                [[[3.0], [4.0]]],
                [[[5.0], [6.0]]],
            ]
        )
        gathered = gather_paged_kv_torch(
            pages,
            block_tables=[[0, 1], [2]],
            sequence_lengths=[3, 2],
            block_size_tokens=2,
            output_length=4,
        )
        expected = torch.tensor(
            [
                [[[0.0], [1.0], [2.0], [3.0]]],
                [[[0.0], [0.0], [5.0], [6.0]]],
            ]
        )
        self.assertTrue(torch.equal(gathered, expected))

    def test_wrapper_matches_reference(self) -> None:
        pages = torch.arange(1, 1 + 4 * 1 * 2 * 2, dtype=torch.float32).view(4, 1, 2, 2)
        ref = gather_paged_kv_torch(
            pages,
            block_tables=[[0, 1], [2, 3]],
            sequence_lengths=[3, 4],
            block_size_tokens=2,
            output_length=5,
        )
        result = gather_paged_kv(
            pages,
            block_tables=[[0, 1], [2, 3]],
            sequence_lengths=[3, 4],
            block_size_tokens=2,
            output_length=5,
            use_triton=False,
        )
        self.assertTrue(torch.equal(ref, result))

    @unittest.skipUnless(torch.cuda.is_available() and TRITON_PAGED_KV_RUNTIME_USABLE, "CUDA Triton path unavailable")
    def test_triton_matches_reference(self) -> None:
        pages = torch.arange(1, 1 + 8 * 2 * 4 * 8, dtype=torch.float16, device="cuda").view(8, 2, 4, 8)
        block_tables = [[0, 1, 2], [3, 4]]
        sequence_lengths = [10, 6]
        ref = gather_paged_kv_torch(
            pages,
            block_tables=block_tables,
            sequence_lengths=sequence_lengths,
            block_size_tokens=4,
            output_length=12,
        )
        result = gather_paged_kv(
            pages,
            block_tables=block_tables,
            sequence_lengths=sequence_lengths,
            block_size_tokens=4,
            output_length=12,
            use_triton=True,
        )
        self.assertTrue(torch.equal(ref, result))


if __name__ == "__main__":
    unittest.main()
