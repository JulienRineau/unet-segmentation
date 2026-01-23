import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from dataset import (
    ADE20KDataConfig,
    ADE20KDataModule,
    ADE20K_IGNORE_INDEX,
    discover_ade20k_test_root,
    discover_ade20k_trainval_root,
)


class TestLocalSplits(unittest.TestCase):
    def _write_image(self, path: Path) -> None:
        array = np.zeros((4, 4, 3), dtype=np.uint8)
        Image.fromarray(array).save(path)

    def _write_mask(self, path: Path) -> None:
        array = np.array([[0, 1, 2, 0], [2, 1, 0, 2], [1, 1, 2, 0], [0, 2, 1, 1]], dtype=np.uint8)
        Image.fromarray(array).save(path)

    def test_split_discovery_and_io(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_images = root / "ADEChallengeData2016" / "images" / "training"
            val_images = root / "ADEChallengeData2016" / "images" / "validation"
            train_masks = root / "ADEChallengeData2016" / "annotations" / "training"
            val_masks = root / "ADEChallengeData2016" / "annotations" / "validation"
            test_images = root / "release_test" / "testing"

            for path in [train_images, val_images, train_masks, val_masks, test_images]:
                path.mkdir(parents=True, exist_ok=True)

            self._write_image(train_images / "train_0001.jpg")
            self._write_mask(train_masks / "train_0001.png")
            self._write_image(val_images / "val_0001.jpg")
            self._write_mask(val_masks / "val_0001.png")
            self._write_image(test_images / "test_0001.jpg")

            self.assertEqual(discover_ade20k_trainval_root(root), root / "ADEChallengeData2016")
            self.assertEqual(discover_ade20k_test_root(root), root / "release_test")

            config = ADE20KDataConfig(
                data_root=str(root),
                image_size=4,
                train_subset=1,
                val_subset=1,
                test_subset=1,
            )
            datamodule = ADE20KDataModule(config)
            datamodule.setup()

            val_image, val_mask = datamodule.val_dataset[0]
            self.assertEqual(val_image.shape[0], 3)
            self.assertEqual(val_mask.shape[-2:], (4, 4))
            self.assertIn(ADE20K_IGNORE_INDEX, val_mask.unique().tolist())

            test_item = datamodule.test_dataset[0]
            self.assertEqual(len(test_item), 2)
            self.assertIsInstance(test_item[1], str)


if __name__ == "__main__":
    unittest.main()
