import unittest

import numpy as np

from veo_watermark_remover.lama_cpu import (
    composite_mask_only,
    context_crop_box,
    mask_in_context,
)


class LamaCpuGeometryTest(unittest.TestCase):
    def test_bottom_right_context_contains_logo(self) -> None:
        crop = context_crop_box(1920, 1080, (1847, 1027, 55, 27), 256)
        self.assertEqual((crop.x, crop.y, crop.width, crop.height), (1664, 824, 256, 256))

    def test_context_size_must_be_modulo_eight(self) -> None:
        with self.assertRaises(ValueError):
            context_crop_box(1920, 1080, (1847, 1027, 55, 27), 250)

    def test_maps_roi_mask_into_context(self) -> None:
        roi_mask = np.zeros((76, 96), dtype=np.uint8)
        roi_mask[20, 30] = 255
        crop = context_crop_box(1920, 1080, (1847, 1027, 55, 27), 256)
        result = mask_in_context(roi_mask, (1824, 1004, 96, 76), crop)
        self.assertEqual(result[1004 - 824 + 20, 1824 - 1664 + 30], 255)
        self.assertEqual(np.count_nonzero(result), 1)

    def test_composite_changes_only_mask(self) -> None:
        original = np.full((8, 8, 3), 20, dtype=np.uint8)
        prediction = np.full_like(original, 220)
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[3:5, 3:5] = 255
        result = composite_mask_only(original, prediction, mask)
        self.assertTrue(np.all(result[mask == 0] == 20))
        self.assertTrue(np.all(result[mask > 0] == 220))


if __name__ == "__main__":
    unittest.main()
