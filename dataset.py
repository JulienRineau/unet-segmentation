import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ADE20K_IGNORE_INDEX = 255


def map_ade20k_labels(mask: torch.Tensor, ignore_index: int = ADE20K_IGNORE_INDEX) -> torch.Tensor:
    if mask.ndim == 3:
        mask = mask[0]
    mask = mask.to(torch.int64)
    ignore = mask == 0
    mask = mask - 1
    mask[ignore] = ignore_index
    return mask


def make_color_palette(num_classes: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(num_classes, 3), dtype=np.uint8)


def colorize_mask(
    mask: torch.Tensor, palette: np.ndarray, ignore_index: int = ADE20K_IGNORE_INDEX
) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    if mask.ndim == 3:
        mask = mask.squeeze(0)
    h, w = mask.shape
    output = np.zeros((h, w, 3), dtype=np.uint8)
    valid = mask != ignore_index
    output[valid] = palette[mask[valid]]
    return output


@dataclass
class ADE20KDataConfig:
    image_size: int = 512
    train_batch_size: int = 8
    val_batch_size: int = 8
    num_workers: int = 8
    min_scale: float = 0.5
    max_scale: float = 2.0
    hflip_prob: float = 0.5
    ignore_index: int = ADE20K_IGNORE_INDEX
    data_root: Optional[str] = None
    cache_dir: Optional[str] = None
    train_subset: Optional[int] = None
    val_subset: Optional[int] = None
    test_subset: Optional[int] = None
    trust_remote_code: bool = False
    train_resize_only: bool = False
    val_on_train: bool = False


def _iter_named_dirs(base_dir: Path, name: str, max_depth: int = 4) -> list[Path]:
    matches = []
    base_dir = base_dir.resolve()
    for root, dirs, _ in os.walk(base_dir):
        current = Path(root)
        depth = len(current.relative_to(base_dir).parts)
        if depth > max_depth:
            dirs[:] = []
            continue
        if current.name == name:
            matches.append(current)
    return matches


def _has_trainval_layout(root: Path) -> bool:
    return (
        (root / "images" / "training").is_dir()
        and (root / "images" / "validation").is_dir()
        and (root / "annotations" / "training").is_dir()
        and (root / "annotations" / "validation").is_dir()
    )


def _has_test_layout(root: Path) -> bool:
    return _get_test_images_dir(root) is not None


def _get_test_images_dir(root: Path) -> Optional[Path]:
    images_dir = root / "images"
    if images_dir.is_dir():
        return images_dir
    testing_dir = root / "testing"
    if testing_dir.is_dir():
        return testing_dir
    return None


def discover_ade20k_trainval_root(data_root: str | Path) -> Path:
    base = Path(data_root)
    candidates = [base, base / "ADEChallengeData2016"]
    candidates.extend(_iter_named_dirs(base, "ADEChallengeData2016"))
    for candidate in candidates:
        if candidate.is_dir() and _has_trainval_layout(candidate):
            return candidate
    raise FileNotFoundError(
        "Could not find ADE20K train/val layout. Expected images/training, images/validation, "
        "annotations/training, annotations/validation under an ADEChallengeData2016 directory."
    )


def discover_ade20k_test_root(data_root: str | Path) -> Path:
    base = Path(data_root)
    candidates = [base, base / "release_test"]
    candidates.extend(_iter_named_dirs(base, "release_test"))
    for candidate in candidates:
        if candidate.is_dir() and _has_test_layout(candidate):
            return candidate
    raise FileNotFoundError(
        "Could not find ADE20K test layout. Expected release_test/images or release_test/testing "
        "under the data root."
    )


def _list_image_files(directory: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    return sorted([p for p in directory.iterdir() if p.suffix.lower() in exts])


def _build_image_mask_pairs(image_dir: Path, mask_dir: Path) -> tuple[list[Path], list[Path]]:
    image_paths = _list_image_files(image_dir)
    mask_paths = []
    missing = []
    for image_path in image_paths:
        mask_path = mask_dir / f"{image_path.stem}.png"
        if not mask_path.exists():
            missing.append(mask_path.name)
        mask_paths.append(mask_path)
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} mask files in {mask_dir}. Example: {missing[0]}"
        )
    return image_paths, mask_paths


class ADE20KLocalDataset(Dataset):
    def __init__(
        self,
        image_paths: list[Path],
        mask_paths: Optional[list[Path]],
        config: ADE20KDataConfig,
        is_train: bool,
        return_filename: bool = False,
    ):
        super().__init__()
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.config = config
        self.is_train = is_train
        self.return_filename = return_filename
        self.normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def _resize(
        self, image: Image.Image, mask: Optional[Image.Image]
    ) -> tuple[Image.Image, Optional[Image.Image]]:
        size = (self.config.image_size, self.config.image_size)
        image = TF.resize(image, size=size, interpolation=Image.BILINEAR)
        if mask is not None:
            mask = TF.resize(mask, size=size, interpolation=Image.NEAREST)
        return image, mask

    def _random_scale_and_crop(
        self, image: Image.Image, mask: Optional[Image.Image]
    ) -> tuple[Image.Image, Optional[Image.Image]]:
        scale = random.uniform(self.config.min_scale, self.config.max_scale)
        new_h = max(1, int(image.height * scale))
        new_w = max(1, int(image.width * scale))
        image = TF.resize(image, size=(new_h, new_w), interpolation=Image.BILINEAR)
        if mask is not None:
            mask = TF.resize(mask, size=(new_h, new_w), interpolation=Image.NEAREST)

        target_h = self.config.image_size
        target_w = self.config.image_size
        pad_h = max(0, target_h - new_h)
        pad_w = max(0, target_w - new_w)
        if pad_h > 0 or pad_w > 0:
            image = TF.pad(image, padding=(0, 0, pad_w, pad_h), fill=0)
            if mask is not None:
                mask = TF.pad(mask, padding=(0, 0, pad_w, pad_h), fill=0)

        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(target_h, target_w)
        )
        image = TF.crop(image, i, j, h, w)
        if mask is not None:
            mask = TF.crop(mask, i, j, h, w)
        return image, mask

    def _maybe_hflip(
        self, image: Image.Image, mask: Optional[Image.Image]
    ) -> tuple[Image.Image, Optional[Image.Image]]:
        if random.random() < self.config.hflip_prob:
            image = TF.hflip(image)
            if mask is not None:
                mask = TF.hflip(mask)
        return image, mask

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        mask = None
        if self.mask_paths is not None:
            mask = Image.open(self.mask_paths[index])

        if self.is_train and not self.config.train_resize_only:
            image, mask = self._random_scale_and_crop(image, mask)
            image, mask = self._maybe_hflip(image, mask)
        else:
            image, mask = self._resize(image, mask)

        image_tensor = TF.pil_to_tensor(image).float() / 255.0
        image_tensor = self.normalize(image_tensor)

        if mask is None:
            if self.return_filename:
                return image_tensor, image_path.name
            return image_tensor

        mask_tensor = TF.pil_to_tensor(mask)
        mask_tensor = map_ade20k_labels(mask_tensor, ignore_index=self.config.ignore_index)
        return image_tensor, mask_tensor


class ADE20KDataset(Dataset):
    def __init__(self, dataset, config: ADE20KDataConfig, is_train: bool):
        super().__init__()
        self.dataset = dataset
        self.config = config
        self.is_train = is_train
        self.normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def _resize(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        size = (self.config.image_size, self.config.image_size)
        image = TF.resize(image, size=size, interpolation=Image.BILINEAR)
        mask = TF.resize(mask, size=size, interpolation=Image.NEAREST)
        return image, mask

    def _random_scale_and_crop(
        self, image: Image.Image, mask: Image.Image
    ) -> tuple[Image.Image, Image.Image]:
        scale = random.uniform(self.config.min_scale, self.config.max_scale)
        new_h = max(1, int(image.height * scale))
        new_w = max(1, int(image.width * scale))
        image = TF.resize(image, size=(new_h, new_w), interpolation=Image.BILINEAR)
        mask = TF.resize(mask, size=(new_h, new_w), interpolation=Image.NEAREST)

        target_h = self.config.image_size
        target_w = self.config.image_size
        pad_h = max(0, target_h - new_h)
        pad_w = max(0, target_w - new_w)
        if pad_h > 0 or pad_w > 0:
            image = TF.pad(image, padding=(0, 0, pad_w, pad_h), fill=0)
            mask = TF.pad(mask, padding=(0, 0, pad_w, pad_h), fill=0)

        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(target_h, target_w)
        )
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)
        return image, mask

    def _maybe_hflip(
        self, image: Image.Image, mask: Image.Image
    ) -> tuple[Image.Image, Image.Image]:
        if random.random() < self.config.hflip_prob:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        return image, mask

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.dataset[index]
        image = item["image"].convert("RGB")
        mask = item["annotation"]

        if self.is_train and not self.config.train_resize_only:
            image, mask = self._random_scale_and_crop(image, mask)
            image, mask = self._maybe_hflip(image, mask)
        else:
            image, mask = self._resize(image, mask)

        image = TF.pil_to_tensor(image).float() / 255.0
        image = self.normalize(image)
        mask = TF.pil_to_tensor(mask)
        mask = map_ade20k_labels(mask, ignore_index=self.config.ignore_index)
        return image, mask


class ADE20KDataModule(pl.LightningDataModule):
    def __init__(self, config: ADE20KDataConfig):
        super().__init__()
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.config.data_root:
            trainval_root = discover_ade20k_trainval_root(self.config.data_root)
            train_images = trainval_root / "images" / "training"
            train_masks = trainval_root / "annotations" / "training"
            val_images = trainval_root / "images" / "validation"
            val_masks = trainval_root / "annotations" / "validation"

            train_image_paths, train_mask_paths = _build_image_mask_pairs(
                train_images, train_masks
            )
            val_image_paths, val_mask_paths = _build_image_mask_pairs(val_images, val_masks)

            if self.config.train_subset:
                train_image_paths = train_image_paths[: self.config.train_subset]
                train_mask_paths = train_mask_paths[: self.config.train_subset]
            if self.config.val_subset:
                val_image_paths = val_image_paths[: self.config.val_subset]
                val_mask_paths = val_mask_paths[: self.config.val_subset]

            self.train_dataset = ADE20KLocalDataset(
                train_image_paths, train_mask_paths, self.config, is_train=True
            )
            self.val_dataset = ADE20KLocalDataset(
                val_image_paths, val_mask_paths, self.config, is_train=False
            )
            if self.config.val_on_train:
                self.val_dataset = self.train_dataset

            try:
                test_root = discover_ade20k_test_root(self.config.data_root)
                test_images = _get_test_images_dir(test_root)
                if test_images is None:
                    raise FileNotFoundError(
                        "Could not find ADE20K test images. Expected release_test/images or "
                        "release_test/testing."
                    )
                test_image_paths = _list_image_files(test_images)
                if self.config.test_subset:
                    test_image_paths = test_image_paths[: self.config.test_subset]
                self.test_dataset = ADE20KLocalDataset(
                    test_image_paths,
                    mask_paths=None,
                    config=self.config,
                    is_train=False,
                    return_filename=True,
                )
            except FileNotFoundError:
                self.test_dataset = None
        else:
            train = load_dataset(
                "scene_parse_150",
                split="train",
                cache_dir=self.config.cache_dir,
                trust_remote_code=self.config.trust_remote_code,
            )
            val = load_dataset(
                "scene_parse_150",
                split="validation",
                cache_dir=self.config.cache_dir,
                trust_remote_code=self.config.trust_remote_code,
            )

            if self.config.train_subset:
                train = train.select(range(self.config.train_subset))
            if self.config.val_subset:
                val = val.select(range(self.config.val_subset))

            self.train_dataset = ADE20KDataset(train, self.config, is_train=True)
            self.val_dataset = ADE20KDataset(val, self.config, is_train=False)
            if self.config.val_on_train:
                self.val_dataset = self.train_dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=self.config.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.val_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=self.config.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise ValueError(
                "Test split not found. Set --data-root to a directory containing release_test."
            )
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.val_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=self.config.num_workers > 0,
        )
