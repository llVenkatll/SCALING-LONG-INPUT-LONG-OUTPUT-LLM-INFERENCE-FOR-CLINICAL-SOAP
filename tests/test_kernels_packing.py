import unittest

import torch

from clinical_speech.kernels.packing import (
    TRITON_RUNTIME_USABLE,
    pack_left_padded_sequences,
    pack_left_padded_sequences_torch,
)


class PackingKernelTest(unittest.TestCase):
    def test_torch_reference_left_pads_sequences(self) -> None:
        sequences = [
            torch.tensor([1, 2, 3], dtype=torch.long),
            torch.tensor([4, 5], dtype=torch.long),
        ]
        ids, mask = pack_left_padded_sequences_torch(
            sequences,
            pad_token_id=0,
            device=torch.device("cpu"),
            pad_to_multiple_of=4,
        )
        self.assertTrue(torch.equal(ids, torch.tensor([[0, 1, 2, 3], [0, 0, 4, 5]])))
        self.assertTrue(torch.equal(mask, torch.tensor([[0, 1, 1, 1], [0, 0, 1, 1]])))

    def test_wrapper_matches_reference_when_triton_disabled(self) -> None:
        sequences = [
            torch.tensor([10, 11, 12], dtype=torch.long),
            torch.tensor([13], dtype=torch.long),
        ]
        ref_ids, ref_mask = pack_left_padded_sequences_torch(
            sequences,
            pad_token_id=0,
            device=torch.device("cpu"),
        )
        ids, mask = pack_left_padded_sequences(
            sequences,
            pad_token_id=0,
            device=torch.device("cpu"),
            use_triton=False,
        )
        self.assertTrue(torch.equal(ids, ref_ids))
        self.assertTrue(torch.equal(mask, ref_mask))

    @unittest.skipUnless(torch.cuda.is_available() and TRITON_RUNTIME_USABLE, "CUDA Triton path unavailable")
    def test_triton_path_matches_reference(self) -> None:
        sequences = [
            torch.tensor([1, 2, 3, 4], dtype=torch.long),
            torch.tensor([5, 6], dtype=torch.long),
            torch.tensor([7], dtype=torch.long),
        ]
        ref_ids, ref_mask = pack_left_padded_sequences_torch(
            sequences,
            pad_token_id=0,
            device=torch.device("cuda"),
            pad_to_multiple_of=8,
        )
        ids, mask = pack_left_padded_sequences(
            sequences,
            pad_token_id=0,
            device=torch.device("cuda"),
            pad_to_multiple_of=8,
            use_triton=True,
        )
        self.assertTrue(torch.equal(ids.cpu(), ref_ids.cpu()))
        self.assertTrue(torch.equal(mask.cpu(), ref_mask.cpu()))
