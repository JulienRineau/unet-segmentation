import logging
from dataclasses import dataclass
import pytorch_lightning as pl
import torch
from datasets import load_dataset
from pytorch_lightning.loggers import WandbLogger
from torch.nn import functional as F
from torch.utils.data import DataLoader
import wandb
from dataset import HuggingFacePILImageDataset, custom_collate
from unet import UNET, UnetConfig


class UNETLightning(pl.LightningModule):
    def __init__(
        self,
        model_config: UnetConfig,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
    ):
        super().__init__()
        self.model = UNET(model_config)
        self.num_classes = model_config.out_channels
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.save_hyperparameters()
        self.log_every_n_steps = 20

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, batch_idx):
        image, mask = batch
        logits = self(image)
        mask = mask.squeeze(1)  # Remove channel dimension if present
        loss = F.cross_entropy(logits, mask)
        return loss, logits, mask

    def training_step(self, batch, batch_idx):
        loss, logits, mask = self._shared_step(batch, batch_idx)
        pred = logits.argmax(dim=1)
        accuracy = (pred == mask).float().mean()

        self.log("train_loss_step", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log(
            "train_accuracy_step", accuracy, on_step=True, on_epoch=False, prog_bar=True
        )
        self.log("learning_rate", self.learning_rate, on_step=True, on_epoch=False)
        self.log("train_loss_epoch", loss, on_step=False, on_epoch=True)
        self.log("train_accuracy_epoch", accuracy, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, mask = self._shared_step(batch, batch_idx)
        pred = logits.argmax(dim=1)
        accuracy = (pred == mask).float().mean()

        self.log("val_loss_step", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log(
            "val_accuracy_step", accuracy, on_step=True, on_epoch=False, prog_bar=True
        )
        self.log("val_loss_epoch", loss, on_step=False, on_epoch=True)
        self.log("val_accuracy_epoch", accuracy, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        return optimizer

    def train_dataloader(self):
        train_dataset = load_dataset(
            "scene_parse_150",
            "instance_segmentation",
            split="train[:100]",  # Use only first 100 samples
        )
        train_dataset = HuggingFacePILImageDataset(train_dataset)
        return DataLoader(
            train_dataset,
            batch_size=16,
            num_workers=4,
            shuffle=True,
            collate_fn=custom_collate,
        )

    def val_dataloader(self):
        val_dataset = load_dataset(
            "scene_parse_150",
            "instance_segmentation",
            split="validation[:20]",  # Use only first 20 samples
        )
        val_dataset = HuggingFacePILImageDataset(val_dataset)
        return DataLoader(
            val_dataset, batch_size=16, num_workers=4, collate_fn=custom_collate
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    pl.seed_everything(42)
    wandb_logger = WandbLogger(project="U-Net-overfit", log_model="all")
    model = UNETLightning(UnetConfig())
    trainer = pl.Trainer(
        max_epochs=200,  # Increase epochs for overfitting
        accelerator="auto",
        devices="auto",
        precision="16-mixed",
        logger=wandb_logger,
        gradient_clip_val=1.0,
        log_every_n_steps=1,
        overfit_batches=1.0,  # Overfit on the entire (small) dataset
    )
    trainer.fit(model)
    wandb.finish()
