import unittest

import torch

from dataset import ADE20K_IGNORE_INDEX
from segmentation_metrics import compute_per_image_iou


class TestRenderOverlaysIoUFilter(unittest.TestCase):
    def test_min_iou_filter_keeps_good_sample(self):
        num_classes = 3

        pred_good = torch.tensor([[0, 1], [2, 2]])
        target_good = torch.tensor([[0, 1], [ADE20K_IGNORE_INDEX, 2]])
        pred_bad = torch.tensor([[0, 0], [0, 0]])
        target_bad = torch.tensor([[1, 1], [1, 1]])

        good_iou = compute_per_image_iou(
            pred_good, target_good, num_classes, ADE20K_IGNORE_INDEX
        )
        bad_iou = compute_per_image_iou(
            pred_bad, target_bad, num_classes, ADE20K_IGNORE_INDEX
        )

        self.assertGreaterEqual(good_iou, 0.99)
        self.assertLessEqual(bad_iou, 0.01)

        min_iou = 0.5
        kept = [iou for iou in (good_iou, bad_iou) if iou >= min_iou]
        self.assertEqual(len(kept), 1)


if __name__ == "__main__":
    unittest.main()
