"""
Run all experiment configs in experiments/configs/*.yaml sequentially.

Usage (from repo root):
    PYTHONPATH=src python experiments/sweep.py [--models knn rf] [--force] [--configs <glob>]

Each config generates its own training CSV under data/simulated/<exp>/ and trains
both models, storing results under data/results/<exp>/{knn,rf}/.

A comparison table is printed at the end and saved to data/results/sweep_summary.csv.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = Path(__file__).resolve().parent / "configs"
PYTHON = sys.executable


def run(cmd: list[str], label: str) -> bool:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    elapsed = time.time() - t0
    ok = result.returncode == 0
    status = "OK" if ok else f"FAILED (exit {result.returncode})"
    print(f"  → {status}  ({elapsed:.1f}s)")
    return ok


def load_metrics(config_path: Path, model: str) -> dict | None:
    exp = config_path.stem
    if exp == "baseline_room":
        metrics_path = REPO_ROOT / "data" / "results" / model / "metrics.json"
    else:
        metrics_path = REPO_ROOT / "data" / "results" / exp / model / "metrics.json"
    if metrics_path.is_file():
        return json.loads(metrics_path.read_text())
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="Sweep all experiment configs")
    p.add_argument("--models", nargs="+", choices=("knn", "rf", "mlp"), default=["knn", "rf", "mlp"])
    p.add_argument(
        "--simulator",
        choices=("pathloss", "sionna"),
        default="pathloss",
        help="RSSI source for CSV generation (default: pathloss)",
    )
    p.add_argument("--force", action="store_true", help="Regenerate CSV even if it exists")
    p.add_argument(
        "--configs",
        nargs="*",
        default=None,
        metavar="NAME",
        help="Config stems to run (default: all). E.g.: corners_4gw_12x8 random_4gw_12x8",
    )
    args = p.parse_args()

    all_configs = sorted(CONFIGS_DIR.glob("*.yaml"))
    if args.configs:
        all_configs = [c for c in all_configs if c.stem in args.configs]

    if not all_configs:
        sys.exit(f"No configs found in {CONFIGS_DIR}")

    print(f"\nExperiments to run: {[c.stem for c in all_configs]}")
    print(f"Models: {args.models}")

    results: list[dict] = []
    base_cmd = [PYTHON, "-m", "ble_indoor"]

    for cfg in all_configs:
        exp = cfg.stem
        gen_cmd = base_cmd + ["generate-csv", "--config", str(cfg), "--simulator", args.simulator]
        if args.force:
            gen_cmd.append("--force")
        ok = run(gen_cmd, f"[{exp}] generate-csv")
        if not ok:
            print(f"  Skipping training for {exp} (CSV generation failed)")
            continue

        for model in args.models:
            train_cmd = base_cmd + ["train", "--model", model, "--config", str(cfg), "--no-sweep"]
            ok = run(train_cmd, f"[{exp}] train --model {model}")
            if ok:
                m = load_metrics(cfg, model)
                if m:
                    val = m.get("validation", {})
                    pos = val.get("position", {})
                    zone = val.get("zone", {})
                    results.append({
                        "simulator": args.simulator,
                        "experiment": exp,
                        "model": model,
                        "n_train": m.get("n_train"),
                        "zone_acc": zone.get("accuracy"),
                        "rmse_m": pos.get("rmse_xy_m"),
                        "mean_m": pos.get("mean_m"),
                        "p90_m": pos.get("p90_m"),
                    })

    if not results:
        print("\nNo results collected.")
        return

    # Print comparison table
    print(f"\n{'='*60}")
    print("  SWEEP SUMMARY")
    print(f"{'='*60}")
    header = f"{'simulator':<10} {'experiment':<30} {'model':<5} {'zone_acc':>9} {'rmse_m':>8} {'mean_m':>8} {'p90_m':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        zone_acc = f"{r['zone_acc']:.3f}" if r["zone_acc"] is not None else "   —"
        rmse = f"{r['rmse_m']:.3f}" if r["rmse_m"] is not None else "   —"
        mean = f"{r['mean_m']:.3f}" if r["mean_m"] is not None else "   —"
        p90 = f"{r['p90_m']:.3f}" if r["p90_m"] is not None else "   —"
        print(f"{r['simulator']:<10} {r['experiment']:<30} {r['model']:<5} {zone_acc:>9} {rmse:>8} {mean:>8} {p90:>8}")

    # Save CSV summary — merge with any existing rows for experiments not in this run
    import csv
    summary_path = REPO_ROOT / "data" / "results" / "sweep_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    run_keys = {(r["simulator"], r["experiment"], r["model"]) for r in results}
    if summary_path.is_file():
        with summary_path.open(newline="") as f:
            for row in csv.DictReader(f):
                key = (row.get("simulator", "pathloss"), row["experiment"], row["model"])
                if key not in run_keys:
                    existing.append(row)
    all_rows = existing + results
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSaved: {summary_path} ({len(all_rows)} rows total)")


if __name__ == "__main__":
    main()
