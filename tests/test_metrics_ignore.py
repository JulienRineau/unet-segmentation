import unittest

import torch
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex

from dataset import ADE20K_IGNORE_INDEX


class TestMetricsIgnoreIndex(unittest.TestCase):
    def test_miou_ignores_index(self):
        preds = torch.tensor([[[0, 1], [2, 2]]])
        target = torch.tensor([[[0, 1], [ADE20K_IGNORE_INDEX, 2]]])
        metric = MulticlassJaccardIndex(
            num_classes=3, ignore_index=ADE20K_IGNORE_INDEX
        )
        metric.update(preds, target)
        miou = metric.compute().item()
        self.assertAlmostEqual(miou, 1.0, places=6)

    def test_metrics_perfect_prediction(self):
        preds = torch.tensor([[[0, 1], [2, 2]]])
        target = torch.tensor([[[0, 1], [2, ADE20K_IGNORE_INDEX]]])
        miou = MulticlassJaccardIndex(
            num_classes=3, ignore_index=ADE20K_IGNORE_INDEX
        )
        mean_acc = MulticlassAccuracy(
            num_classes=3, average="macro", ignore_index=ADE20K_IGNORE_INDEX
        )
        pixel_acc = MulticlassAccuracy(
            num_classes=3, average="micro", ignore_index=ADE20K_IGNORE_INDEX
        )
        miou.update(preds, target)
        mean_acc.update(preds, target)
        pixel_acc.update(preds, target)
        self.assertAlmostEqual(miou.compute().item(), 1.0, places=6)
        self.assertAlmostEqual(mean_acc.compute().item(), 1.0, places=6)
        self.assertAlmostEqual(pixel_acc.compute().item(), 1.0, places=6)

    def test_miou_ignores_absent_classes(self):
        preds = torch.tensor([[[0, 0], [1, 1]]])
        target = torch.tensor([[[0, 1], [0, 1]]])
        metric = MulticlassJaccardIndex(num_classes=3)
        metric.update(preds, target)
        miou = metric.compute().item()
        expected = (1.0 / 3.0 + 1.0 / 3.0) / 2.0
        self.assertAlmostEqual(miou, expected, places=6)


if __name__ == "__main__":
    unittest.main()
