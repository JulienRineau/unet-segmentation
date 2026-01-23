import argparse
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex

from backbones import DEFAULT_ENCODER_NAME, default_weights_path, get_encoder_spec
from dataset import (
    ADE20KDataConfig,
    ADE20KDataModule,
    ADE20K_IGNORE_INDEX,
    colorize_mask,
    make_color_palette,
)
from unet import UNET, UnetConfig


@dataclass
class TrainConfig:
    learning_rate: float = 3e-4
    encoder_lr_mult: float = 1.0
    weight_decay: float = 0.01
    max_epochs: int = 80
    max_steps: int = -1
    precision: str = "bf16-mixed"
    accelerator: str = "gpu"
    devices: int = 8
    strategy: str = "ddp"
    accumulate_grad_batches: int = 1
    gradient_clip_val: float = 1.0
    seed: int = 42
    log_every_n_steps: int = 50
    check_val_every_n_epoch: int = 1
    sync_batchnorm: bool = False
    log_images: bool = True
    log_image_count: int = 4
    log_image_every_n_epochs: int = 1


class SegmentationLitModule(pl.LightningModule):
    def __init__(
        self,
        model_config: UnetConfig | dict,
        train_config: TrainConfig | dict,
        ignore_index: int = ADE20K_IGNORE_INDEX,
    ):
        super().__init__()
        if isinstance(model_config, dict):
            model_config = UnetConfig(**model_config)
        if isinstance(train_config, dict):
            train_config = TrainConfig(**train_config)
        self.model = UNET(model_config)
        if train_config.sync_batchnorm:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        self.save_hyperparameters(
            {
                "model_config": asdict(model_config),
                "train_config": asdict(train_config),
                "ignore_index": ignore_index,
            }
        )
        self.ignore_index = ignore_index
        self.num_classes = model_config.out_channels
        self.train_config = train_config
        self.palette = make_color_palette(self.num_classes)

        self.val_miou = MulticlassJaccardIndex(
            num_classes=self.num_classes, ignore_index=self.ignore_index
        )
        self.val_mean_acc = MulticlassAccuracy(
            num_classes=self.num_classes, average="macro", ignore_index=self.ignore_index
        )
        self.val_pixel_acc = MulticlassAccuracy(
            num_classes=self.num_classes, average="micro", ignore_index=self.ignore_index
        )
        self.example_batch = None
        self.test_example_batch = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        image, mask = batch
        logits = self(image)
        loss = F.cross_entropy(logits, mask, ignore_index=self.ignore_index)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        image, mask = batch
        logits = self(image)
        loss = F.cross_entropy(logits, mask, ignore_index=self.ignore_index)
        preds = torch.argmax(logits, dim=1)
        self.val_miou.update(preds, mask)
        self.val_mean_acc.update(preds, mask)
        self.val_pixel_acc.update(preds, mask)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        if batch_idx == 0 and self.trainer.is_global_zero:
            self.example_batch = (
                image[: self.train_config.log_image_count].detach(),
                mask[: self.train_config.log_image_count].detach(),
                preds[: self.train_config.log_image_count].detach(),
            )

    def on_validation_epoch_end(self) -> None:
        self.log(
            "val_miou",
            self.val_miou.compute(),
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "val_mean_acc",
            self.val_mean_acc.compute(),
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "val_pixel_acc",
            self.val_pixel_acc.compute(),
            prog_bar=False,
            sync_dist=True,
        )
        self.val_miou.reset()
        self.val_mean_acc.reset()
        self.val_pixel_acc.reset()

        if (
            self.train_config.log_images
            and self.trainer.is_global_zero
            and self.example_batch is not None
            and self.current_epoch % self.train_config.log_image_every_n_epochs == 0
        ):
            self._log_images(*self.example_batch)
            self._log_test_preview()
        self.example_batch = None

    def _log_test_preview(self) -> None:
        if not self.trainer or not self.trainer.is_global_zero:
            return
        if not self.train_config.log_images:
            return
        if self.current_epoch % self.train_config.log_image_every_n_epochs != 0:
            return

        datamodule = self.trainer.datamodule
        if datamodule is None:
            return

        count = self.train_config.log_image_count
        images = None
        filenames = None

        test_dataset = getattr(datamodule, "test_dataset", None)
        if test_dataset is not None:
            try:
                length = len(test_dataset)
            except TypeError:
                length = 0
            if length > 0:
                count = min(count, length)
                image_items: list[torch.Tensor] = []
                name_items: list[str] = []
                for idx in range(count):
                    sample = test_dataset[idx]
                    if isinstance(sample, (list, tuple)) and len(sample) == 2:
                        image_tensor, filename = sample
                        image_items.append(image_tensor)
                        name_items.append(str(filename))
                    else:
                        image_items.append(sample)
                images = torch.stack(image_items, dim=0)
                filenames = name_items or None

        if images is None:
            try:
                test_loader = datamodule.test_dataloader()
                batch = next(iter(test_loader))
            except Exception:
                return

            if isinstance(batch, (list, tuple)) and len(batch) >= 1:
                images = batch[0]
                if len(batch) > 1:
                    filenames = batch[1]
            else:
                images = batch

        if images is None:
            return

        if isinstance(filenames, (list, tuple)):
            filenames = [str(name) for name in list(filenames)[:count]]
        elif filenames is not None:
            filenames = [str(filenames)]

        images = images[:count].to(self.device, non_blocking=True)

        was_training = self.training
        self.eval()
        with torch.no_grad():
            logits = self(images)
            preds = torch.argmax(logits, dim=1)
        if was_training:
            self.train()

        self._log_test_images(images.detach(), preds.detach(), filenames)

    def test_step(self, batch, batch_idx):
        images = batch[0]
        filenames = None
        if isinstance(batch, (list, tuple)) and len(batch) > 1:
            filenames = batch[1]
        logits = self(images)
        preds = torch.argmax(logits, dim=1)

        if batch_idx == 0 and self.trainer.is_global_zero:
            if isinstance(filenames, (list, tuple)):
                filenames = list(filenames[: self.train_config.log_image_count])
            self.test_example_batch = (
                images[: self.train_config.log_image_count].detach(),
                preds[: self.train_config.log_image_count].detach(),
                filenames,
            )

    def on_test_epoch_end(self) -> None:
        if (
            self.train_config.log_images
            and self.trainer.is_global_zero
            and self.test_example_batch is not None
        ):
            self._log_test_images(*self.test_example_batch)
        self.test_example_batch = None

    def _log_images(self, images, masks, preds) -> None:
        if not isinstance(self.logger, WandbLogger):
            return
        import wandb

        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
        images = (images * std + mean).clamp(0, 1)
        images = images.permute(0, 2, 3, 1).cpu().numpy()
        images = (images * 255).astype(np.uint8)

        panels = []
        for idx in range(images.shape[0]):
            gt_color = colorize_mask(masks[idx], self.palette, self.ignore_index)
            pred_color = colorize_mask(preds[idx], self.palette, self.ignore_index)
            panel = np.concatenate([images[idx], gt_color, pred_color], axis=1)
            panels.append(panel)

        full_panel = np.concatenate(panels, axis=0)
        self.logger.experiment.log(
            {"val_samples": wandb.Image(full_panel, caption="image | gt | pred")},
            step=self.global_step,
        )

    def _log_test_images(self, images, preds, filenames) -> None:
        if not isinstance(self.logger, WandbLogger):
            return
        import wandb

        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
        images = (images * std + mean).clamp(0, 1)
        images = images.permute(0, 2, 3, 1).cpu().numpy()
        images = (images * 255).astype(np.uint8)

        panels = []
        captions = []
        for idx in range(images.shape[0]):
            pred_color = colorize_mask(preds[idx], self.palette, self.ignore_index)
            panel = np.concatenate([images[idx], pred_color], axis=1)
            panels.append(panel)
            if filenames:
                captions.append(f"test:{filenames[idx]}")

        full_panel = np.concatenate(panels, axis=0)
        payload = {"test_samples": wandb.Image(full_panel, caption="image | pred")}
        if captions:
            payload["test_filenames"] = captions
        self.logger.experiment.log(payload, step=self.global_step)

    def configure_optimizers(self):
        encoder_params = []
        decoder_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith(("downs.", "bottleneck.", "encoder.")):
                encoder_params.append(param)
            else:
                decoder_params.append(param)

        if not encoder_params or self.train_config.encoder_lr_mult == 1.0:
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.train_config.learning_rate,
                weight_decay=self.train_config.weight_decay,
            )
        else:
            optimizer = torch.optim.AdamW(
                [
                    {
                        "params": encoder_params,
                        "lr": self.train_config.learning_rate
                        * self.train_config.encoder_lr_mult,
                    },
                    {"params": decoder_params, "lr": self.train_config.learning_rate},
                ],
                weight_decay=self.train_config.weight_decay,
            )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs
            if self.trainer and self.trainer.max_epochs > 0
            else self.train_config.max_epochs,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


class SyntheticSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples: int, image_size: int, num_classes: int, ignore_index: int):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int):
        image = torch.rand(3, self.image_size, self.image_size)
        mask = torch.randint(0, self.num_classes, (self.image_size, self.image_size))
        if index % 2 == 0:
            mask[0:8, 0:8] = self.ignore_index
        return image, mask


class SyntheticDataModule(pl.LightningDataModule):
    def __init__(self, image_size: int, num_classes: int, ignore_index: int):
        super().__init__()
        self.image_size = image_size
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = SyntheticSegmentationDataset(
            num_samples=4,
            image_size=self.image_size,
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
        )
        self.val_dataset = SyntheticSegmentationDataset(
            num_samples=2,
            image_size=self.image_size,
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=2, shuffle=True)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ADE20K U-Net DDP baseline training.")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--val-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--min-scale", type=float, default=0.5)
    parser.add_argument("--max-scale", type=float, default=2.0)
    parser.add_argument("--hflip-prob", type=float, default=0.5)
    parser.add_argument("--train-resize-only", action="store_true")
    parser.add_argument("--val-on-train", action="store_true")
    parser.add_argument("--ignore-index", type=int, default=ADE20K_IGNORE_INDEX)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--train-subset", type=int, default=None)
    parser.add_argument("--val-subset", type=int, default=None)
    parser.add_argument("--test-subset", type=int, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--encoder-name", type=str, default=DEFAULT_ENCODER_NAME)
    parser.add_argument("--encoder-weights", type=str, default=None)
    parser.add_argument(
        "--use-pretrained-encoder",
        dest="use_pretrained_encoder",
        action="store_true",
    )
    parser.add_argument(
        "--no-pretrained-encoder",
        dest="use_pretrained_encoder",
        action="store_false",
    )
    parser.set_defaults(use_pretrained_encoder=True)

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--encoder-lr-mult", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--devices", type=int, default=8)
    parser.add_argument("--strategy", type=str, default="ddp")
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every-n-steps", type=int, default=50)
    parser.add_argument("--check-val-every-n-epoch", type=int, default=1)
    parser.add_argument("--sync-batchnorm", action="store_true")
    parser.add_argument("--log-images", dest="log_images", action="store_true")
    parser.add_argument("--no-log-images", dest="log_images", action="store_false")
    parser.set_defaults(log_images=True)
    parser.add_argument("--log-image-count", type=int, default=4)
    parser.add_argument("--log-image-every-n-epochs", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--disable-wandb", action="store_true")

    parser.add_argument("--wandb-project", type=str, default="ade20k-unet-ddp")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def resolve_pretrained_encoder(
    encoder_name: str,
    encoder_weights: str | None,
    use_pretrained_encoder: bool,
) -> tuple[bool, str | None]:
    if not use_pretrained_encoder:
        return False, None
    get_encoder_spec(encoder_name)
    weights_path = Path(encoder_weights) if encoder_weights else default_weights_path(encoder_name)
    if not weights_path.is_file():
        logging.warning(
            "Pretrained weights not found at %s. Falling back to plain U-Net. "
            "Run `python scripts/download_pretrained_weights.py --encoder-name %s`.",
            weights_path,
            encoder_name,
        )
        return False, None
    return True, str(weights_path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    pl.seed_everything(args.seed, workers=True)

    encoder_name = args.encoder_name
    try:
        use_pretrained_encoder, encoder_weights_path = resolve_pretrained_encoder(
            encoder_name=encoder_name,
            encoder_weights=args.encoder_weights,
            use_pretrained_encoder=args.use_pretrained_encoder,
        )
    except ValueError as exc:
        raise SystemExit(str(exc))

    model_config = UnetConfig(
        out_channels=150,
        use_pretrained_encoder=use_pretrained_encoder,
        encoder_name=encoder_name if use_pretrained_encoder else None,
        encoder_weights_path=str(encoder_weights_path) if encoder_weights_path else None,
    )
    data_config = ADE20KDataConfig(
        image_size=args.image_size,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
        hflip_prob=args.hflip_prob,
        ignore_index=args.ignore_index,
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        train_subset=args.train_subset,
        val_subset=args.val_subset,
        test_subset=args.test_subset,
        trust_remote_code=args.trust_remote_code,
        train_resize_only=args.train_resize_only,
        val_on_train=args.val_on_train,
    )
    train_config = TrainConfig(
        learning_rate=args.learning_rate,
        encoder_lr_mult=args.encoder_lr_mult,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        precision=args.precision,
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        seed=args.seed,
        log_every_n_steps=args.log_every_n_steps,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        sync_batchnorm=args.sync_batchnorm,
        log_images=args.log_images,
        log_image_count=args.log_image_count,
        log_image_every_n_epochs=args.log_image_every_n_epochs,
    )

    logger = False
    is_rank_zero = int(os.environ.get("LOCAL_RANK", "0")) == 0
    if not args.disable_wandb:
        run_name = args.run_name
        if not run_name:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            run_name = f"ade20k-unet-{timestamp}"
        logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            log_model=False,
            save_dir=os.getcwd(),
        )
        if is_rank_zero:
            logger.log_hyperparams(
                {
                    "model": asdict(model_config),
                    "data": asdict(data_config),
                    "train": asdict(train_config),
                }
            )

    callbacks = [
        ModelCheckpoint(
            monitor="val_miou",
            mode="max",
            save_top_k=1,
            save_last=True,
            filename="ade20k-unet-{epoch:03d}-{val_miou:.4f}",
        ),
    ]
    if logger:
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    trainer = pl.Trainer(
        max_epochs=train_config.max_epochs,
        max_steps=train_config.max_steps,
        accelerator=train_config.accelerator,
        devices=train_config.devices,
        strategy=train_config.strategy,
        precision=train_config.precision,
        accumulate_grad_batches=train_config.accumulate_grad_batches,
        gradient_clip_val=train_config.gradient_clip_val,
        log_every_n_steps=train_config.log_every_n_steps,
        check_val_every_n_epoch=train_config.check_val_every_n_epoch,
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=True,
    )

    if args.dry_run:
        datamodule = SyntheticDataModule(
            image_size=args.image_size,
            num_classes=model_config.out_channels,
            ignore_index=data_config.ignore_index,
        )
    else:
        datamodule = ADE20KDataModule(data_config)

    model = SegmentationLitModule(
        model_config=model_config,
        train_config=train_config,
        ignore_index=data_config.ignore_index,
    )
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
