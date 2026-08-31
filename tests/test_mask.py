import unittest

import cv2
import numpy as np

from veo_watermark_remover.mask import MaskCalibrationError, build_shape_mask


class ShapeMaskTest(unittest.TestCase):
    def test_keeps_three_letters_and_never_fills_bbox(self) -> None:
        image = np.zeros((30, 50, 3), dtype=np.uint8)
        cv2.rectangle(image, (12, 12), (14, 20), (230, 230, 230), -1)
        cv2.rectangle(image, (20, 14), (24, 20), (230, 230, 230), -1)
        cv2.rectangle(image, (30, 14), (35, 20), (230, 230, 230), -1)
        result = build_shape_mask(image, (100, 100, 50, 30), (110, 110, 30, 12), dilation=1)
        self.assertEqual(result.component_count, 3)
        self.assertEqual(result.detected_component_count, 3)
        self.assertEqual(result.selection_strategy, "three_largest_components")
        self.assertGreaterEqual(result.confidence, 0.62)
        self.assertGreaterEqual(result.final_pixel_count, result.raw_pixel_count)
        self.assertEqual(result.effective_dilation, 1)
        self.assertLess(result.final_pixel_count, 30 * 12)

    def test_small_logo_growth_is_constrained_by_video_evidence(self) -> None:
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        cv2.rectangle(image, (5, 7), (7, 14), (230, 230, 230), -1)
        cv2.rectangle(image, (11, 9), (14, 14), (230, 230, 230), -1)
        cv2.rectangle(image, (18, 9), (22, 14), (230, 230, 230), -1)
        result = build_shape_mask(
            image, (100, 100, 30, 20), (104, 106, 20, 9), dilation=1
        )
        self.assertEqual(result.requested_dilation, 1)
        self.assertEqual(result.effective_dilation, 1)
        self.assertEqual(result.final_pixel_count, result.raw_pixel_count)

    def test_hysteresis_adds_observed_antialias_neighbor(self) -> None:
        image = np.zeros((30, 50, 3), dtype=np.uint8)
        cv2.rectangle(image, (12, 12), (14, 20), (230, 230, 230), -1)
        cv2.rectangle(image, (20, 14), (24, 20), (230, 230, 230), -1)
        cv2.rectangle(image, (30, 14), (35, 20), (230, 230, 230), -1)
        image[14, 19] = (80, 80, 80)
        result = build_shape_mask(
            image, (100, 100, 50, 30), (110, 110, 30, 12), dilation=1
        )
        self.assertEqual(result.raw_mask[14, 19], 0)
        self.assertEqual(result.mask[14, 19], 255)

    def test_rejects_bbox_outside_roi(self) -> None:
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            build_shape_mask(image, (0, 0, 20, 20), (19, 19, 5, 5))

    def test_accepts_three_letters_joined_by_compression_bridge(self) -> None:
        image = np.zeros((30, 50, 3), dtype=np.uint8)
        cv2.rectangle(image, (12, 12), (14, 20), (230, 230, 230), -1)
        cv2.rectangle(image, (20, 14), (24, 20), (230, 230, 230), -1)
        cv2.rectangle(image, (30, 14), (35, 20), (230, 230, 230), -1)
        cv2.line(image, (14, 17), (20, 17), (230, 230, 230), 1)
        cv2.line(image, (24, 17), (30, 17), (230, 230, 230), 1)
        result = build_shape_mask(
            image, (100, 100, 50, 30), (110, 110, 30, 12), dilation=1
        )
        self.assertEqual(result.detected_component_count, 1)
        self.assertEqual(result.selection_strategy, "adaptive_merged_components")
        self.assertLess(result.final_pixel_count, 30 * 12)

    def test_rejects_merged_foreground_that_fills_logo_bbox(self) -> None:
        image = np.zeros((30, 50, 3), dtype=np.uint8)
        cv2.rectangle(image, (10, 10), (39, 21), (230, 230, 230), -1)
        with self.assertRaisesRegex(MaskCalibrationError, "calibration rejected"):
            build_shape_mask(
                image, (100, 100, 50, 30), (110, 110, 30, 12), dilation=1
            )

    def test_does_not_use_template_as_final_mask_on_bright_background(self) -> None:
        image = np.zeros((30, 50, 3), dtype=np.uint8)
        cv2.rectangle(image, (10, 10), (39, 21), (230, 230, 230), -1)
        cv2.rectangle(image, (10, 10), (13, 21), (0, 0, 0), -1)
        with self.assertRaises(MaskCalibrationError) as raised:
            build_shape_mask(
                image, (100, 100, 50, 30), (110, 110, 30, 12), dilation=0
            )
        self.assertGreater(raised.exception.details["coverage_ratio"], 0.70)


if __name__ == "__main__":
    unittest.main()
