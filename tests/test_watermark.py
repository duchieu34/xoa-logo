import unittest

from veo_watermark_remover.watermark import load_measurement, scaled_logo_bbox


class WatermarkMeasurementTest(unittest.TestCase):
    def test_reference_bbox_scales_to_1080p(self) -> None:
        self.assertEqual(scaled_logo_bbox(1920, 1080, load_measurement()), (1864, 1042, 32, 14))


if __name__ == "__main__":
    unittest.main()
