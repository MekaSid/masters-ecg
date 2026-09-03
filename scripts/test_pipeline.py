from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.beat_extraction import BeatWindow, extract_beats
from src.data.load_ecg import load_annotations, load_noise_record, load_record
from src.data.noise import NoiseAugmentor
from src.tda.distances import compute_diagram_distance
from src.tda.pipeline import BeatTDAPipeline, TakensEmbeddingConfig, VietorisRipsConfig
from src.tda.representations import PersistenceImageConfig
from src.utils.config import load_project_configs
from src.utils.paths import REPO_ROOT as PROJECT_ROOT, ensure_dir
from src.utils.plotting import (
    save_diagram_plot,
    save_embedding_plot,
    save_persistence_image_plot,
    save_waveform_plot,
)
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end smoke test for ECG + noise + TDA pipeline.")
    parser.add_argument("--record", default="100", help="MIT-BIH record ID.")
    parser.add_argument("--beat-index", type=int, default=0, help="Extracted beat index.")
    parser.add_argument("--noise-type", default="ma", choices=["bw", "ma", "em"], help="NSTDB noise type.")
    parser.add_argument("--snr-db", type=float, default=12.0, help="Target SNR in dB.")
    return parser.parse_args()


def select_beat(record_id: str, beat_index: int, configs: dict) -> tuple:
    data_cfg = configs["data"]
    beat_cfg = data_cfg["data"]["beat_window"]
    window = BeatWindow(pre_samples=beat_cfg["pre_samples"], post_samples=beat_cfg["post_samples"])

    mitdb_dir = PROJECT_ROOT / data_cfg["paths"]["mitdb_dir"]
    record = load_record(mitdb_dir, record_id)
    annotations = load_annotations(mitdb_dir, record_id)
    beats = extract_beats(record.signal, annotations.samples, annotations.symbols, record_id, window)
    if not beats:
        raise RuntimeError(f"No beats extracted from record {record_id}.")
    if beat_index >= len(beats):
        raise IndexError(f"Beat index {beat_index} out of range for record {record_id}; found {len(beats)} beats.")
    return record, beats[beat_index]


def main() -> None:
    args = parse_args()
    configs = load_project_configs()
    set_seed(configs["data"]["project"]["seed"])

    data_cfg = configs["data"]
    tda_cfg = configs["tda"]
    output_dir = ensure_dir(PROJECT_ROOT / data_cfg["paths"]["results_dir"] / "smoke_test")

    record, beat = select_beat(args.record, args.beat_index, configs)
    noise_record = load_noise_record(PROJECT_ROOT / data_cfg["paths"]["nstdb_dir"], args.noise_type)
    augmentor = NoiseAugmentor(seed=data_cfg["noise"]["seed"])
    noisy_result = augmentor.add_noise(beat.waveform, noise_record.signal, args.noise_type, args.snr_db)

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
    clean_output = pipeline.run_vietoris_rips(noisy_result.clean)
    noisy_output = pipeline.run_vietoris_rips(noisy_result.noisy)

    distances = []
    for dim in clean_output.diagrams:
        distances.append(
            compute_diagram_distance(
                clean_output.diagrams[dim],
                noisy_output.diagrams[dim],
                homology_dimension=dim,
            )
        )

    stem = f"{args.record}_beat{args.beat_index}_{args.noise_type}_{int(args.snr_db)}db"
    save_waveform_plot(noisy_result.clean, noisy_result.noisy, output_dir / f"{stem}_waveforms.png", f"Record {args.record} beat {args.beat_index}")
    save_embedding_plot(clean_output.embedding, output_dir / f"{stem}_clean_embedding.png", "Clean delay embedding")
    save_embedding_plot(noisy_output.embedding, output_dir / f"{stem}_noisy_embedding.png", "Noisy delay embedding")
    save_diagram_plot(clean_output.diagrams, output_dir / f"{stem}_clean_diagrams.png", "Clean persistence diagrams")
    save_diagram_plot(noisy_output.diagrams, output_dir / f"{stem}_noisy_diagrams.png", "Noisy persistence diagrams")

    for image_result in clean_output.persistence_images.values():
        save_persistence_image_plot(
            image_result.image,
            output_dir / f"{stem}_clean_H{image_result.homology_dimension}_pi.png",
            f"Clean persistence image H{image_result.homology_dimension}",
        )
    for image_result in noisy_output.persistence_images.values():
        save_persistence_image_plot(
            image_result.image,
            output_dir / f"{stem}_noisy_H{image_result.homology_dimension}_pi.png",
            f"Noisy persistence image H{image_result.homology_dimension}",
        )

    print(f"Record: {record.record_id} (fs={record.fs} Hz)")
    print(f"Beat index: {beat.beat_index}, original symbol: {beat.original_symbol}, mapped class: {beat.mapped_class}")
    print(f"Noise type: {args.noise_type}, target SNR: {args.snr_db:.2f} dB, achieved SNR: {noisy_result.snr_db_achieved:.2f} dB")
    for metric in distances:
        print(
            f"H{metric.homology_dimension}: "
            f"bottleneck={metric.bottleneck_distance:.6f}, "
            f"wasserstein={metric.wasserstein_distance:.6f}"
        )
    for dim, stats in clean_output.statistics.items():
        print(
            f"Clean H{dim} stats: "
            f"features={stats.num_features}, total_persistence={stats.total_persistence:.6f}, "
            f"max_persistence={stats.max_persistence:.6f}, entropy={stats.persistent_entropy:.6f}"
        )
    print(f"Diagnostic plots saved to {Path(output_dir)}")


if __name__ == "__main__":
    main()
