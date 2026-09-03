from __future__ import annotations

import argparse

import numpy as np

from src.data.beat_extraction import BeatWindow, extract_beats
from src.data.load_ecg import load_annotations, load_record
from src.tda.embedding import delay_embed
from src.tda.persistence import compute_persistence_diagrams
from src.tda.representations import build_persistence_image
from src.utils.config import load_project_configs
from src.utils.paths import REPO_ROOT, ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute persistence diagrams and images for one ECG beat.")
    parser.add_argument("--record", default="100", help="MIT-BIH record ID.")
    parser.add_argument("--beat-index", type=int, default=0, help="Index within extracted beats.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs = load_project_configs()
    data_cfg = configs["data"]
    tda_cfg = configs["tda"]

    window_cfg = data_cfg["data"]["beat_window"]
    window = BeatWindow(pre_samples=window_cfg["pre_samples"], post_samples=window_cfg["post_samples"])

    record = load_record(REPO_ROOT / data_cfg["paths"]["mitdb_dir"], args.record)
    ann = load_annotations(REPO_ROOT / data_cfg["paths"]["mitdb_dir"], args.record)
    beat = extract_beats(record.signal, ann.samples, ann.symbols, args.record, window)[args.beat_index]

    embedding = delay_embed(
        beat.waveform,
        dimension=tda_cfg["tda"]["embedding_dimension"],
        delay=tda_cfg["tda"]["delay"],
        stride=tda_cfg["tda"]["stride"],
    )
    persistence = compute_persistence_diagrams(
        embedding,
        maxdim=tda_cfg["tda"]["max_homology_dimension"],
        thresh=tda_cfg["tda"]["thresh"],
    )

    image_cfg = tda_cfg["persistence_image"]
    output_dir = ensure_dir(REPO_ROOT / data_cfg["paths"]["tda_diagrams_dir"])
    image_dir = ensure_dir(REPO_ROOT / data_cfg["paths"]["persistence_images_dir"])

    for dim, diagram in enumerate(persistence.diagrams):
        np.save(output_dir / f"{args.record}_beat{args.beat_index}_H{dim}.npy", diagram)
        image = build_persistence_image(
            diagram=diagram,
            homology_dimension=dim,
            pixel_size=image_cfg["pixel_size"],
            birth_range=tuple(image_cfg["birth_range"]),
            pers_range=tuple(image_cfg["pers_range"]),
        )
        np.save(image_dir / f"{args.record}_beat{args.beat_index}_H{dim}_pi.npy", image.image)

    print(f"Computed persistence diagrams for record {args.record}, beat {args.beat_index}.")


if __name__ == "__main__":
    main()
