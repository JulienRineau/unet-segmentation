import unittest

import torch

from dataset import ADE20K_IGNORE_INDEX, map_ade20k_labels


class TestLabelMapping(unittest.TestCase):
    def test_reduce_zero_label(self):
        mask = torch.tensor([[[0, 1, 2, 150]]], dtype=torch.uint8)
        mapped = map_ade20k_labels(mask, ignore_index=ADE20K_IGNORE_INDEX)
        expected = torch.tensor([[ADE20K_IGNORE_INDEX, 0, 1, 149]], dtype=torch.int64)
        self.assertTrue(torch.equal(mapped, expected))

    def test_mapping_range(self):
        mask = torch.arange(0, 151, dtype=torch.uint8).view(1, 1, -1)
        mapped = map_ade20k_labels(mask, ignore_index=ADE20K_IGNORE_INDEX)
        valid = mapped[mapped != ADE20K_IGNORE_INDEX]
        self.assertEqual(valid.min().item(), 0)
        self.assertEqual(valid.max().item(), 149)


if __name__ == "__main__":
    unittest.main()
