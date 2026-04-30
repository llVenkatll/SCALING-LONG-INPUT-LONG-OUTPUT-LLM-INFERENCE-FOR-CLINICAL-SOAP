import unittest

from clinical_speech.runtime.scheduler import PendingRequest, QueueAdmissionScheduler, RequestQueue, StaticBatchScheduler


class StaticBatchSchedulerTest(unittest.TestCase):
    def test_scheduler_groups_prompts_in_fixed_batches(self) -> None:
        scheduler = StaticBatchScheduler(max_batch_size=3)
        batches = scheduler.schedule([f"prompt-{idx}" for idx in range(7)])
        self.assertEqual([len(batch.prompts) for batch in batches], [3, 3, 1])
        self.assertEqual(batches[0].request_ids[0], "req_00000")
        self.assertEqual(batches[2].request_ids[-1], "req_00006")

    def test_queue_admission_is_fcfs_and_capacity_bound(self) -> None:
        queue = RequestQueue()
        queue.push(PendingRequest(request_id="req_a", prompt="a"))
        queue.push(PendingRequest(request_id="req_b", prompt="b"))
        queue.push(PendingRequest(request_id="req_c", prompt="c"))

        scheduler = QueueAdmissionScheduler(max_batch_size=2, max_concurrent_requests=3)
        first = scheduler.admit(queue, active_requests=1)
        self.assertEqual([item.request_id for item in first], ["req_a", "req_b"])
        second = scheduler.admit(queue, active_requests=2)
        self.assertEqual([item.request_id for item in second], ["req_c"])
        self.assertEqual(len(queue), 0)
