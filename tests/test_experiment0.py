import unittest

from veo_watermark_remover.experiment0 import select_frame_indices


class FrameSelectionTest(unittest.TestCase):
    def test_samples_whole_timeline(self) -> None:
        result = select_frame_indices(101, 6)
        self.assertEqual(result[0], 0)
        self.assertEqual(result[-1], 100)
        self.assertEqual(len(result), 6)

    def test_does_not_duplicate_short_video_frames(self) -> None:
        self.assertEqual(select_frame_indices(3, 8), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()

