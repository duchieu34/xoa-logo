import unittest

import numpy as np

from veo_watermark_remover.optical_flow import (
    BILATERAL_FLOW_CODE,
    reconstruct_optical_flow_temporal,
)


class OpticalFlowTemporalTest(unittest.TestCase):
    def _stack(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        original = np.full((3, 13, 15, 3), 30, dtype=np.uint8)
        for frame in range(3):
            original[frame, 0, 0] = frame
        original[0, 6, 2] = 80
        original[2, 6, 8] = 80
        alpha = original.copy()
        mask = np.zeros((13, 15), dtype=np.uint8)
        mask[6, 5] = 255
        valid = np.zeros((3, 13, 15), dtype=bool)
        confidence = np.ones((13, 15), dtype=np.float32)
        return original, alpha, valid, mask, confidence

    @staticmethod
    def _translation_estimator(
        target: np.ndarray, source: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        displacement = float(int(source[0, 0, 0]) - int(target[0, 0, 0])) * 3.0
        forward = np.zeros((*mask.shape, 2), dtype=np.float32)
        backward = np.zeros_like(forward)
        forward[..., 0] = displacement
        backward[..., 0] = -displacement
        return forward, backward

    def test_uses_only_a_clean_warped_source(self) -> None:
        original, alpha, valid, mask, confidence = self._stack()
        result = reconstruct_optical_flow_temporal(
            original, alpha, valid, mask, confidence,
            offsets=(-1, 1), single_confidence_min=0.0,
            bilateral_confidence_min=0.0, flow_estimator=self._translation_estimator,
        )
        self.assertEqual(result.donor_source[1, 6, 5], BILATERAL_FLOW_CODE)
        self.assertTrue(np.array_equal(result.restored_stack[1, 6, 5], [80, 80, 80]))

    def test_never_copies_an_unresolved_warped_source(self) -> None:
        original, alpha, valid, mask, confidence = self._stack()

        def zero_flow(target: np.ndarray, source: np.ndarray, source_mask: np.ndarray):
            flow = np.zeros((*source_mask.shape, 2), dtype=np.float32)
            return flow, flow.copy()

        result = reconstruct_optical_flow_temporal(
            original, alpha, valid, mask, confidence,
            offsets=(-1, 1), single_confidence_min=0.0,
            bilateral_confidence_min=0.0, flow_estimator=zero_flow,
        )
        self.assertEqual(result.donor_mask[:, 6, 5].sum(), 0)
        self.assertTrue(np.all(result.unresolved_stack[:, 6, 5] == 255))

    def test_rejects_forward_backward_inconsistency(self) -> None:
        original, alpha, valid, mask, confidence = self._stack()

        def inconsistent(target: np.ndarray, source: np.ndarray, source_mask: np.ndarray):
            forward = np.zeros((*source_mask.shape, 2), dtype=np.float32)
            backward = np.zeros_like(forward)
            forward[..., 0] = 3
            backward[..., 0] = 3
            return forward, backward

        result = reconstruct_optical_flow_temporal(
            original, alpha, valid, mask, confidence,
            offsets=(-1, 1), single_confidence_min=0.0,
            bilateral_confidence_min=0.0, flow_estimator=inconsistent,
        )
        self.assertEqual(result.donor_mask[:, 6, 5].sum(), 0)


if __name__ == "__main__":
    unittest.main()
