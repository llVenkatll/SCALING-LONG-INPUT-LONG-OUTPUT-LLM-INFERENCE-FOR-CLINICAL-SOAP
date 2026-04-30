import time
import unittest

import torch

from clinical_speech.models.note_generator import FirstStepTimingProcessor, FirstTokenTimingStreamer


class TimingUtilityTests(unittest.TestCase):
    def test_first_step_processor_records_first_timestamp(self) -> None:
        processor = FirstStepTimingProcessor(synchronize_cuda=False)
        self.assertIsNone(processor.first_step_time)
        before = time.perf_counter()
        processor(torch.tensor([[1, 2, 3]]), torch.zeros((1, 5)))
        self.assertIsNotNone(processor.first_step_time)
        self.assertGreaterEqual(processor.first_step_time, before)

    def test_streamer_skips_prompt_and_records_first_generated_token(self) -> None:
        streamer = FirstTokenTimingStreamer(skip_prompt=True, synchronize_cuda=False)
        streamer.put(torch.tensor([[10, 11, 12]]))
        self.assertIsNone(streamer.first_token_time)

        time.sleep(0.01)
        before = time.perf_counter()
        streamer.put(torch.tensor([[13]]))
        self.assertIsNotNone(streamer.first_token_time)
        self.assertGreaterEqual(streamer.first_token_time, before)


if __name__ == "__main__":
    unittest.main()
