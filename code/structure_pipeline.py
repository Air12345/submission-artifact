from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def comparable_structure_config(structure):
    operational = {"state_save_freq", "tf32", "tna_query_cache", "reuse_forward_true_score"}
    return {key: value for key, value in structure.items() if key not in operational}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, default=PACKAGE_ROOT / "data")
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = args.run_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = args.data_root.resolve()
    metadata = {"dataset": config["dataset"], "seed": args.seed, "config": config,
                "effective_structure": config["structure"], "data_root": str(source)}
    saved = output / "RUN_CONFIG.json"
    if saved.exists() and json.loads(saved.read_text()) != metadata:
        raise ValueError("Existing structural run has a different configuration")
    saved.write_text(json.dumps(metadata, indent=2))
    env = dict(os.environ, TSMC_DATA_ROOT=str(source), PYTHONHASHSEED=str(args.seed),
               PYTHONDONTWRITEBYTECODE="1")
    directory = output / "stage1"
    checkpoint = directory / "best_structure.pt"
    marker = directory / "DONE.json"
    if not marker.exists():
        subprocess.run([sys.executable, "-B", str(PACKAGE_ROOT / "code/train_structure.py"),
                        "--config", str(args.config.resolve()), "--dataset_view", config["dataset"],
                        "--run_dir", str(directory), "--seed", str(args.seed), "--gpu", str(args.gpu)],
                       check=True, env=env)
        marker.write_text(json.dumps({"sha256": file_sha256(checkpoint)}, indent=2))
    elif json.loads(marker.read_text())["sha256"] != file_sha256(checkpoint):
        raise ValueError("Completed structural checkpoint hash mismatch")
    report = {"checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint),
              "data_root": str(source)}
    (output / "STRUCTURE_REPORT.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
