import unittest

import cv2
import numpy as np

from veo_watermark_remover.mask import build_shape_mask


class ShapeMaskTest(unittest.TestCase):
    def test_keeps_three_letters_and_never_fills_bbox(self) -> None:
        image = np.zeros((30, 50, 3), dtype=np.uint8)
        cv2.rectangle(image, (12, 12), (14, 20), (230, 230, 230), -1)
        cv2.rectangle(image, (20, 14), (24, 20), (230, 230, 230), -1)
        cv2.rectangle(image, (30, 14), (35, 20), (230, 230, 230), -1)
        result = build_shape_mask(image, (100, 100, 50, 30), (110, 110, 30, 12), dilation=1)
        self.assertEqual(result.component_count, 3)
        self.assertGreater(result.final_pixel_count, result.raw_pixel_count)
        self.assertLess(result.final_pixel_count, 30 * 12)

    def test_rejects_bbox_outside_roi(self) -> None:
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            build_shape_mask(image, (0, 0, 20, 20), (19, 19, 5, 5))


if __name__ == "__main__":
    unittest.main()
