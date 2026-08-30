import unittest

from veo_watermark_remover.config import DEFAULT_CANDIDATE_ROI, RelativeROI


class RelativeROITest(unittest.TestCase):
    def test_scales_to_resolution(self) -> None:
        self.assertEqual(RelativeROI(0.5, 0.5, 0.25, 0.25).to_pixels(1920, 1080), (960, 540, 480, 270))

    def test_rejects_out_of_frame_roi(self) -> None:
        with self.assertRaises(ValueError):
            RelativeROI(0.9, 0.9, 0.2, 0.2)

    def test_parses_cli_value(self) -> None:
        self.assertEqual(RelativeROI.parse("0.1,0.2,0.3,0.4"), RelativeROI(0.1, 0.2, 0.3, 0.4))

    def test_default_1080p_context_roi(self) -> None:
        self.assertEqual(DEFAULT_CANDIDATE_ROI.to_pixels(1920, 1080), (1824, 1004, 96, 76))


if __name__ == "__main__":
    unittest.main()
