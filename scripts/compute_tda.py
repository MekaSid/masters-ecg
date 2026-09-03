from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.beat_extraction import BeatWindow, extract_beats
from src.data.load_ecg import load_annotations, load_record
from src.tda.pipeline import BeatTDAPipeline, TakensEmbeddingConfig, VietorisRipsConfig
from src.tda.representations import PersistenceImageConfig
from src.utils.config import load_project_configs
from src.utils.paths import REPO_ROOT as PROJECT_ROOT, ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute persistence diagrams and images for one ECG beat.")
    parser.add_argument("--record", default="100", help="MIT-BIH record ID.")
    parser.add_argument("--beat-index", type=int, default=0, help="Index within extracted beats.")
    parser.add_argument("--method", choices=["takens_vr", "sublevel"], default="takens_vr", help="TDA construction to run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs = load_project_configs()
    data_cfg = configs["data"]
    tda_cfg = configs["tda"]

    window_cfg = data_cfg["data"]["beat_window"]
    window = BeatWindow(pre_samples=window_cfg["pre_samples"], post_samples=window_cfg["post_samples"])

    record = load_record(PROJECT_ROOT / data_cfg["paths"]["mitdb_dir"], args.record)
    ann = load_annotations(PROJECT_ROOT / data_cfg["paths"]["mitdb_dir"], args.record)
    beat = extract_beats(record.signal, ann.samples, ann.symbols, args.record, window)[args.beat_index]

    pipeline = BeatTDAPipeline(
        embedding_config=TakensEmbeddingConfig(
            dimension=tda_cfg["tda"]["embedding_dimension"],
            delay=tda_cfg["tda"]["delay"],
            stride=tda_cfg["tda"]["stride"],
        ),
        vr_config=VietorisRipsConfig(
            max_homology_dimension=tda_cfg["tda"]["max_homology_dimension"],
            max_edge_length=tda_cfg["tda"]["max_edge_length"],
        ),
        image_config=PersistenceImageConfig(
            resolution=tuple(tda_cfg["persistence_image"]["resolution"]),
            birth_range=tuple(tda_cfg["persistence_image"]["birth_range"]),
            persistence_range=tuple(tda_cfg["persistence_image"]["persistence_range"]),
            bandwidth=tda_cfg["persistence_image"]["bandwidth"],
        ),
    )
    if args.method == "takens_vr":
        output = pipeline.run_vietoris_rips(beat.waveform)
    else:
        output = pipeline.run_sublevel(beat.waveform)

    output_dir = ensure_dir(PROJECT_ROOT / data_cfg["paths"]["tda_diagrams_dir"])
    image_dir = ensure_dir(PROJECT_ROOT / data_cfg["paths"]["persistence_images_dir"])

    for dim, diagram in output.diagrams.items():
        np.save(output_dir / f"{args.record}_beat{args.beat_index}_H{dim}.npy", diagram)
        np.save(image_dir / f"{args.record}_beat{args.beat_index}_H{dim}_pi.npy", output.persistence_images[dim].image)

    print(f"Computed {output.method} persistence outputs for record {args.record}, beat {args.beat_index}.")


if __name__ == "__main__":
    main()
