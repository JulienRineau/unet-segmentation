import logging
from dataclasses import dataclass

import pytorch_lightning as pl
import torch
from datasets import load_dataset
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.nn import functional as F

import wandb
from dataset import HuggingFacePILImageDataset
from unet import UNET, UnetConfig


class UNETLightning(pl.LightningModule):
    def __init__(
        self,
        model_config: UnetConfig,
    ):
        super().__init__()
        self.model = UNET(model_config)
        self.save_hyperparameters()

    def forward(self, x):
        logits = self.model(x)

    def _shared_step(self, batch):
        image, mask = batch
        logits = self.model(image)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), mask.view(-1))
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("train_loss", loss)

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            weight_decay=0.1,
            betas=[0.9, 0.95],
            eps=1e-8,
        )
        return optimizer

    def train_dataloader(self):
        logging.info("Loading training dataset...")
        train_dataset = load_dataset(
            "scene_parse_150", "instance_segmentation", split="train", streaming=True
        )
        logging.info(
            "Training dataset loaded. Wrapping with HuggingFacePILImageDataset..."
        )
        train_dataset = HuggingFacePILImageDataset(train_dataset)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=4,
        )
        logging.info("Training DataLoader is ready.")
        return train_dataloader

    def val_dataloader(self):
        logging.info("Loading validation dataset...")
        validation_dataset = load_dataset(
            "scene_parse_150",
            "instance_segmentation",
            split="validation",
            streaming=True,
        )
        logging.info(
            "Validation dataset loaded. Wrapping with HuggingFacePILImageDataset..."
        )
        validation_dataset = HuggingFacePILImageDataset(validation_dataset)
        validation_dataloader = torch.utils.data.DataLoader(
            validation_dataset,
            batch_size=4,
        )
        logging.info("Validation DataLoader is ready.")
        return validation_dataloader


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    torch.set_float32_matmul_precision("high")
    wandb_logger = WandbLogger(project="U-Net")

    model = UNETLightning(UnetConfig())

    trainer = pl.Trainer(
        max_epochs=50,
        precision="bf16-mixed",
        logger=wandb_logger,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="value",
        log_every_n_steps=1,
    )
    trainer.fit(model)

    wandb.finish()
