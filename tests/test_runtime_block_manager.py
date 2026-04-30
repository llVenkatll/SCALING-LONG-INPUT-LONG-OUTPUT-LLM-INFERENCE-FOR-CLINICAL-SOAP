import unittest

from clinical_speech.runtime.block_manager import CacheBudgetExceeded, KVCacheBlockManager, KVCacheLayout


class BlockManagerTest(unittest.TestCase):
    def test_allocate_grow_free_and_reuse_blocks(self) -> None:
        layout = KVCacheLayout(num_layers=2, num_key_value_heads=4, head_dim=8, bytes_per_element=2)
        manager = KVCacheBlockManager(
            layout=layout,
            block_size_tokens=16,
            max_cache_budget_bytes=layout.bytes_per_token * 16 * 4,
        )

        first = manager.allocate("req_a", 10)
        self.assertEqual(first.block_ids, [0])
        manager.grow("req_a", 20)
        self.assertEqual(manager.allocations["req_a"].block_ids, [0, 1])

        second = manager.allocate("req_b", 16)
        self.assertEqual(second.block_ids, [2])
        snapshot = manager.snapshot()
        self.assertEqual(snapshot["used_blocks"], 3)
        self.assertGreater(snapshot["allocated_bytes"], 0)

        manager.free("req_a")
        reused = manager.allocate("req_c", 16)
        self.assertEqual(reused.block_ids[0], 3)

    def test_budget_exceeded_raises(self) -> None:
        layout = KVCacheLayout(num_layers=1, num_key_value_heads=1, head_dim=8, bytes_per_element=2)
        manager = KVCacheBlockManager(
            layout=layout,
            block_size_tokens=8,
            max_cache_budget_bytes=layout.bytes_per_token * 8,
        )
        manager.allocate("req_a", 8)
        with self.assertRaises(CacheBudgetExceeded):
            manager.allocate("req_b", 8)
