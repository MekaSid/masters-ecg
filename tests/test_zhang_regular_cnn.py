import numpy as np
import torch

from src.models.zhang_regular_cnn import ZhangRegularCNN
from src.training.zhang_dataset import build_zhang_feature, dynamic_beat_bounds, rr_ratio_features


def test_dynamic_beat_bounds_match_paper_window() -> None:
    assert dynamic_beat_bounds(100, 400) == (150, 500)


def test_rr_ratios_and_feature_tensor_shape() -> None:
    rpeaks = np.array([100, 300, 500, 800], dtype=np.int64)
    global_ratio, near_ratio = rr_ratio_features(rpeaks)
    assert np.isnan(global_ratio[0])
    assert np.allclose(global_ratio[1:], [200 / 233.333333, 200 / 233.333333, 300 / 233.333333])
    feature = build_zhang_feature(np.arange(256, dtype=np.float32).reshape(128, 2), global_ratio[1], near_ratio[1])
    assert feature.shape == (2, 3, 128)
    assert np.allclose(feature[:, 0, :].mean(axis=1), 0.0)


def test_zhang_regular_cnn_output_shape() -> None:
    model = ZhangRegularCNN()
    output = model(torch.randn(4, 2, 3, 128))
    assert output.shape == (4, 5)
