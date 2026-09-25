#!/usr/bin/env python3
"""Cross-check baseline_*.json sha256 values against the Zenodo archive
(doi:10.5281/zenodo.17095311) file manifest.

Per the mission mandate (item 4): reads sha256 values ALREADY computed in
baseline_M{1,2,3}.json (does not recompute them from gns-sample/); tries a
network call to Zenodo's API first; only falls back to documenting a
blocked state if the network call genuinely fails or the archive is
impractically large to fully fetch and unpack.

Result of running this on 2026-09-24 is recorded in ZENODO_CROSS_CHECK.md
(status: BLOCKED -- see that file for the measured throughput and full
reasoning). This script is still runnable and will re-attempt a live
network check every time; if network conditions improve enough that a
full per-file sha256 diff becomes practical, remove the early-exit at the
bottom and complete the download+extract+diff loop.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
ZENODO_RECORD_ID = "17095311"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"

# Rough archive->model mapping (from the Zenodo file manifest, 2026-09-24 audit).
ARCHIVE_FOR_MODEL = {
    "M1": ["M1.model.rollout.zip", "M1.train.valid.test.zip"],
    "M2": ["M2.model.rollout.zip", "M2.train.valid.test.zip"],
    "M3": [
        "M3.model.rollout.zip",
        "M3.train.valid.test.zip.partaa", "M3.train.valid.test.zip.partab",
        "M3.train.valid.test.zip.partac", "M3.train.valid.test.zip.partad",
        "M3.train.valid.test.zip.partae",
    ],
}


def load_baseline_sha256s():
    out = {}
    for key in ("M1", "M2", "M3"):
        path = HERE / f"baseline_{key}.json"
        if not path.exists():
            continue
        with open(path) as f:
            b = json.load(f)
        out[key] = {
            "checkpoint_sha256": b["checkpoint_sha256"],
            "test_npz_sha256": b["test_npz_sha256"],
        }
    return out


def fetch_manifest(timeout=15):
    resp = requests.get(ZENODO_API_URL, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def measure_throughput(sample_url: str, sample_bytes=20_000_000, timeout=60) -> float:
    """Download a small byte-range and return measured bytes/sec."""
    t0 = time.time()
    headers = {"Range": f"bytes=0-{sample_bytes - 1}"}
    resp = requests.get(sample_url, headers=headers, timeout=timeout, stream=True)
    total = 0
    for chunk in resp.iter_content(chunk_size=1 << 20):
        total += len(chunk)
    elapsed = time.time() - t0
    return total / elapsed if elapsed > 0 else 0.0


def main():
    baselines = load_baseline_sha256s()
    if not baselines:
        print("No baseline_*.json files found -- run extract_baselines.py first.")
        sys.exit(1)

    try:
        manifest = fetch_manifest()
    except Exception as e:
        print(f"BLOCKED: network call to {ZENODO_API_URL} failed: {e}")
        print("See ZENODO_CROSS_CHECK.md.")
        sys.exit(2)

    files = {f["key"]: f for f in manifest.get("files", [])}
    print(f"Fetched Zenodo manifest: {len(files)} files, doi=10.5281/zenodo.{ZENODO_RECORD_ID}")

    # Zenodo exposes MD5 of the packed .zip archives, not sha256 of the
    # individual model.pt/test.npz files inside them (which is what our
    # baseline_*.json records). A real diff requires downloading + unzipping
    # every archive and re-hashing the extracted files with sha256 -- i.e.
    # it cannot be done from the manifest alone. Measure whether that's
    # even practical before attempting it.
    total_bytes = 0
    for key, archive_names in ARCHIVE_FOR_MODEL.items():
        for name in archive_names:
            if name in files:
                total_bytes += files[name]["size"]
    total_gb = total_bytes / 1e9
    print(f"Total archive size needed for a full per-file sha256 diff: {total_gb:.2f} GB")

    sample_key = "M1.model.rollout.zip"
    if sample_key in files:
        sample_url = files[sample_key]["links"]["self"]
        throughput = measure_throughput(sample_url)
        eta_hours = (total_bytes / throughput) / 3600 if throughput > 0 else float("inf")
        print(f"Measured download throughput (20MB sample of {sample_key}): "
              f"{throughput/1e3:.1f} KB/s -> ETA for full {total_gb:.2f} GB: {eta_hours:.1f} h")
        if eta_hours > 0.5:
            print(f"BLOCKED: network/size -- {total_gb:.2f} GB at {throughput/1e3:.1f} KB/s "
                  f"would take ~{eta_hours:.1f} h, impractical for this check. "
                  f"See ZENODO_CROSS_CHECK.md.")
            sys.exit(2)
    else:
        print(f"BLOCKED: expected archive {sample_key!r} not found in manifest "
              f"(manifest may have changed). See ZENODO_CROSS_CHECK.md.")
        sys.exit(2)

    # Only reached if a full download+extract+diff is judged practical.
    print("Proceeding with full archive diff is not implemented in this script "
          "(see module docstring) -- would require download+unzip+sha256 per "
          "archive for M1/M2/M3.")
    sys.exit(2)


if __name__ == "__main__":
    main()
