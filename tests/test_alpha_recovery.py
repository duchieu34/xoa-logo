import unittest

import numpy as np

from veo_watermark_remover.alpha_recovery import (
    deblend_roi,
    estimate_alpha_model,
    estimate_distribution_model,
)


class AlphaRecoveryTest(unittest.TestCase):
    def test_recovers_synthetic_white_watermark(self) -> None:
        frames, height, width = 40, 5, 6
        background = np.empty((frames, height, width, 3), dtype=np.float32)
        for frame in range(frames):
            base = 20 + frame * 4
            background[frame, ..., 0] = base
            background[frame, ..., 1] = base + 8
            background[frame, ..., 2] = base + 15
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[2, 2:5] = 255
        true_alpha = np.array([0.25, 0.45, 0.65], dtype=np.float32)
        observed = background.copy()
        observed[:, 2, 2:5, :] = (
            background[:, 2, 2:5, :] * (1.0 - true_alpha[None, :, None])
            + 248.0 * true_alpha[None, :, None]
        )
        spatial_rmse = np.zeros((height, width), dtype=np.float32)
        model = estimate_alpha_model(
            observed.astype(np.uint8), background, spatial_rmse, mask,
            confidence_threshold=0.1,
        )
        self.assertTrue(np.allclose(model.alpha[2, 2:5], true_alpha, atol=0.04))
        self.assertTrue(np.allclose(model.watermark_bgr, 248.0, atol=5.0))

    def test_unresolved_pixels_are_unchanged(self) -> None:
        from veo_watermark_remover.alpha_recovery import AlphaModel

        shape = (3, 3)
        model = AlphaModel(
            watermark_bgr=np.array([255, 255, 255], dtype=np.float32),
            alpha=np.full(shape, 0.5, dtype=np.float32),
            confidence=np.zeros(shape, dtype=np.float32),
            resolved_mask=np.zeros(shape, dtype=np.uint8),
            transparent_mask=np.zeros(shape, dtype=np.uint8),
            unresolved_mask=np.full(shape, 255, dtype=np.uint8),
            fit_rmse=np.zeros(shape, dtype=np.float32),
            spatial_rmse=np.zeros(shape, dtype=np.float32),
            background_dynamic_range=np.zeros(shape, dtype=np.float32),
            gamut_fraction=np.zeros(shape, dtype=np.float32),
        )
        roi = np.full((3, 3, 3), 123, dtype=np.uint8)
        output, applied = deblend_roi(roi, model)
        self.assertTrue(np.array_equal(output, roi))
        self.assertEqual(int(applied.sum()), 0)

    def test_distribution_model_uses_brightness_variation(self) -> None:
        frames, height, width = 60, 15, 15
        background = np.empty((frames, height, width, 3), dtype=np.float32)
        levels = np.linspace(20, 180, frames, dtype=np.float32)
        for channel, offset in enumerate((0, 5, 10)):
            background[..., channel] = levels[:, None, None] + offset
        observed = background.copy()
        mask = np.zeros((height, width), dtype=np.uint8)
        points = ((6, 6), (7, 7), (8, 8))
        for y, x in points:
            observed[:, y, x, :] = 0.65 * background[:, y, x, :] + 0.35 * 250.0
            mask[y, x] = 255
        model, _ = estimate_distribution_model(
            observed.astype(np.uint8), mask, confidence_threshold=0.1, gamut_min=0.8,
        )
        estimated = np.array([model.alpha[y, x] for y, x in points])
        self.assertTrue(np.allclose(estimated, 0.35, atol=0.05))
        self.assertTrue(np.allclose(model.watermark_bgr, 250.0, atol=6.0))


if __name__ == "__main__":
    unittest.main()
