from __future__ import annotations

import torch

from src.models.raw_model import RawECGConvNet, RawECGModelConfig


def test_raw_ecg_convnet_output_shape() -> None:
    model = RawECGConvNet(RawECGModelConfig(input_length=256, num_classes=3))
    logits = model(torch.randn(4, 1, 256))

    assert logits.shape == (4, 3)
