import unittest

import numpy as np

from veo_watermark_remover.temporal import BILATERAL_DONOR_CODE, reconstruct_direct_temporal


class DirectTemporalTest(unittest.TestCase):
    def _static_stack(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        original = np.full((7, 11, 11, 3), 40, dtype=np.uint8)
        alpha = original.copy()
        mask = np.zeros((11, 11), dtype=np.uint8)
        mask[5, 5] = 255
        valid = np.zeros((7, 11, 11), dtype=bool)
        valid[:, 5, 5] = True
        confidence = np.zeros((11, 11), dtype=np.float32)
        confidence[5, 5] = 0.9
        alpha[:, 5, 5] = 70
        return original, alpha, valid, mask, confidence

    def test_prefers_agreeing_two_sided_donors(self) -> None:
        original, alpha, valid, mask, confidence = self._static_stack()
        valid[3, 5, 5] = False
        alpha[3, 5, 5] = 220
        result = reconstruct_direct_temporal(original, alpha, valid, mask, confidence)
        self.assertEqual(result.donor_source[3, 5, 5], BILATERAL_DONOR_CODE)
        self.assertTrue(np.array_equal(result.restored_stack[3, 5, 5], [70, 70, 70]))

    def test_never_uses_an_unresolved_source(self) -> None:
        original, alpha, valid, mask, confidence = self._static_stack()
        valid[:, 5, 5] = False
        result = reconstruct_direct_temporal(original, alpha, valid, mask, confidence)
        self.assertEqual(result.donor_mask[:, 5, 5].sum(), 0)
        self.assertTrue(np.all(result.unresolved_stack[:, 5, 5] == 255))

    def test_rejects_moving_context(self) -> None:
        original, alpha, valid, mask, confidence = self._static_stack()
        valid[3, 5, 5] = False
        for frame in (0, 1, 2, 4, 5, 6):
            original[frame, 3:8, 3:8] = 220
        result = reconstruct_direct_temporal(original, alpha, valid, mask, confidence)
        self.assertEqual(result.donor_mask[3, 5, 5], 0)


if __name__ == "__main__":
    unittest.main()
