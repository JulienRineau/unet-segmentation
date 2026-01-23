import tempfile
import unittest
from pathlib import Path

import torch

from train import resolve_pretrained_encoder
from unet import UNET, UnetConfig


class TestPretrainedEncoder(unittest.TestCase):
    def test_pretrained_encoder_instantiation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_path = Path(tmpdir) / "missing.pth"
            config = UnetConfig(
                out_channels=3,
                use_pretrained_encoder=True,
                encoder_name="resnet34",
                encoder_weights_path=str(missing_path),
            )
            model = UNET(config)
            self.assertTrue(model.uses_pretrained_encoder)

    def test_pretrained_forward_shape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_path = Path(tmpdir) / "missing.pth"
            config = UnetConfig(
                out_channels=5,
                use_pretrained_encoder=True,
                encoder_name="resnet34",
                encoder_weights_path=str(missing_path),
            )
            model = UNET(config)
            x = torch.randn(2, 3, 64, 64)
            with torch.no_grad():
                preds = model(x)
            self.assertEqual(preds.shape, (2, 5, 64, 64))

    def test_fallback_when_weights_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_path = Path(tmpdir) / "missing.pth"
            use_pretrained, weights_path = resolve_pretrained_encoder(
                encoder_name="resnet34",
                encoder_weights=str(missing_path),
                use_pretrained_encoder=True,
            )
            self.assertFalse(use_pretrained)
            self.assertIsNone(weights_path)


if __name__ == "__main__":
    unittest.main()
