# ADE20K U-Net Segmentation (Lightning DDP)

Baseline semantic segmentation pipeline for ADE20K (scene_parse_150) using a U-Net decoder with either the original U-Net down path or a pretrained ResNet encoder, trained with PyTorch Lightning DDP.

## Setup
Install dependencies:
```
pip install -r requirements.txt
```

## Download pretrained encoder weights (optional)
Download ImageNet-1K pretrained weights into `weights/`:
```
python scripts/download_pretrained_weights.py --encoder-name resnet34
```

Notes:
- Supported encoders: `resnet34`, `resnet50`.
- The download script uses `requests` + `tqdm` (already listed in `requirements.txt`).

## Tests (fast, no ADE20K download)
```
python -m unittest discover -s tests
```

## Debug checks (local ADE20K)
Mask semantics + ignore ratio:
```
python scripts/check_dataset_masks.py --data-root data --split train --num-samples 50
```

Alignment panels (image + GT after transforms):
```
python scripts/save_debug_panels.py --data-root data --split train --num-samples 8 --output-dir debug_panels
```

## Render overlays (val/test)
Generate side-by-side panels (image | gt overlay | pred overlay for val, image | pred overlay for test):
```
python scripts/render_overlays.py \
  --checkpoint /home/ubuntu/Texas/personal/unet-segmentation/checkpoints/ade20k-unet-epoch=079-val_miou=0.3041.ckpt \
  --data-root /home/ubuntu/Texas/personal/unet-segmentation/data \
  --split val \
  --num-samples 8 \
  --output-dir /home/ubuntu/Texas/personal/unet-segmentation/overlays
```
Use `--alpha` to control overlay transparency (0 = image only, 1 = mask only).
Use `--min-iou` to only keep strong val predictions; results are randomly sampled from the matching set (for example: `--min-iou 0.5 --num-samples 8`).
If the default checkpoint is missing, pass your own path (for example, `--checkpoint ade20k-unet-ddp/<run_id>/checkpoints/ade20k-unet-epoch=XXX-val_miou=YYYY.ckpt`).

## Download ADE20K (official MIT zips)
Download and extract the train/val and test zips into `data/`:
```
python scripts/download_ade20k.py --all --data-dir data
```

Expected on-disk layout:
```
data/
  ADEChallengeData2016/
    images/
      training/
      validation/
    annotations/
      training/
      validation/
  release_test/
    testing/
```

## Training (single-node DDP, bf16, W&B)
Example command for 8 GPUs:
```
python train.py \
  --data-root data \
  --devices 8 \
  --accelerator gpu \
  --strategy ddp \
  --precision bf16-mixed \
  --train-batch-size 8 \
  --val-batch-size 8 \
  --num-workers 8 \
  --wandb-project ade20k-unet-ddp
```

Notes:
- By default, training will use the pretrained encoder if weights are available in `weights/`. Run the download script above first.
- Force the original U-Net with `--no-pretrained-encoder`.
- Switch encoders with `--encoder-name resnet50`. Use `--encoder-weights /path/to/weights.pth` to override the default path.
- Use `--data-root data` to train from the local ADE20K layout above.
- If `--data-root` is not set, ADE20K is downloaded on first run via HuggingFace datasets. Use `--cache-dir` to control where it is stored.
- Use `--train-subset` and `--val-subset` for quick debug runs.
- Use `--train-resize-only` to disable train-time random scale/crop/flip and match val transforms.
- Use `--val-on-train` to evaluate on the training subset (overfit/debug).
- If the local test split exists under `data/`, a small fixed panel of test predictions is logged to W&B alongside validation samples; use `--test-subset` and `--log-image-count` to limit what gets logged.
- Disable W&B logging with `--disable-wandb`.
- If the dataset loader prompts for remote code, add `--trust-remote-code`.
- Optional synthetic dry-run (no dataset download):
```
python train.py --dry-run --devices 1 --accelerator cpu --precision 32-true --disable-wandb
```
