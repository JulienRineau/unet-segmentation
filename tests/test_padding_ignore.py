import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from dataset import ADE20KDataConfig, ADE20KLocalDataset, ADE20K_IGNORE_INDEX


class TestPaddingIgnore(unittest.TestCase):
    def test_padding_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_path = root / "sample.jpg"
            mask_path = root / "sample.png"

            image = np.zeros((4, 4, 3), dtype=np.uint8)
            mask = np.ones((4, 4), dtype=np.uint8)
            Image.fromarray(image).save(image_path)
            Image.fromarray(mask).save(mask_path)

            config = ADE20KDataConfig(
                image_size=4,
                min_scale=0.5,
                max_scale=0.5,
                hflip_prob=0.0,
            )
            dataset = ADE20KLocalDataset(
                [image_path], [mask_path], config, is_train=True
            )

            _, mapped = dataset[0]
            ignore_count = int((mapped == ADE20K_IGNORE_INDEX).sum().item())
            self.assertGreater(ignore_count, 0)
            self.assertTrue(
                ((mapped == ADE20K_IGNORE_INDEX) | ((mapped >= 0) & (mapped <= 149))).all()
            )


if __name__ == "__main__":
    unittest.main()
