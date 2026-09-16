import numpy as np
import torch

from src.models.betti_fusion_model import DindinPHOnlyCNN, ZhangBettiFusionCNN
from src.tda.betti import BettiCurveConfig, betti_curve, dindin_betti_curves


def test_betti_curve_counts_barcode_intervals() -> None:
    diagram = np.array([[0.0, 1.0], [0.5, 1.5]], dtype=np.float32)
    curve = betti_curve(diagram, resolution=4, essential_death=1.5)
    assert curve.tolist() == [1.0, 2.0, 2.0, 1.0]


def test_dindin_curves_are_finite_and_fixed_size() -> None:
    signal = np.sin(np.linspace(0, 4 * np.pi, 64, dtype=np.float32))
    curves = dindin_betti_curves(signal, BettiCurveConfig(resolution=32))
    assert curves.shape == (2, 32)
    assert np.isfinite(curves).all()
    assert np.all(curves >= 0)


def test_ph_only_and_fusion_output_shapes() -> None:
    raw = torch.randn(3, 2, 3, 128)
    betti = torch.randn(3, 2, 128)
    assert DindinPHOnlyCNN()(betti).shape == (3, 5)
    assert ZhangBettiFusionCNN()(raw, betti).shape == (3, 5)
