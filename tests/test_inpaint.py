import unittest

import numpy as np

from veo_watermark_remover.inpaint import inpaint_roi


class InpaintTest(unittest.TestCase):
    def test_preserves_pixels_outside_mask(self) -> None:
        image = np.full((20, 20, 3), 80, dtype=np.uint8)
        image[8:12, 8:12] = 255
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[8:12, 8:12] = 255
        result = inpaint_roi(image, mask, "telea", 3)
        self.assertTrue(np.array_equal(result[mask == 0], image[mask == 0]))
        self.assertFalse(np.array_equal(result[mask > 0], image[mask > 0]))

    def test_rejects_unknown_method(self) -> None:
        image = np.zeros((5, 5, 3), dtype=np.uint8)
        mask = np.zeros((5, 5), dtype=np.uint8)
        with self.assertRaises(ValueError):
            inpaint_roi(image, mask, "unknown")


if __name__ == "__main__":
    unittest.main()

