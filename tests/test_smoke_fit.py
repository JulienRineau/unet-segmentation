import unittest

import pytorch_lightning as pl

from train import SegmentationLitModule, SyntheticDataModule, TrainConfig
from unet import UnetConfig


class TestSmokeFit(unittest.TestCase):
    def test_smoke_fit_cpu(self):
        pl.seed_everything(0, workers=True)
        model_config = UnetConfig(out_channels=3)
        train_config = TrainConfig(max_epochs=1, log_images=False)
        model = SegmentationLitModule(model_config, train_config)
        datamodule = SyntheticDataModule(
            image_size=64, num_classes=3, ignore_index=255
        )

        trainer = pl.Trainer(
            max_epochs=1,
            accelerator="cpu",
            devices=1,
            limit_train_batches=1,
            limit_val_batches=1,
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
        )
        trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    unittest.main()
