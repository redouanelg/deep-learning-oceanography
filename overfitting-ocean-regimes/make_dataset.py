#!/usr/bin/env python3
"""Rebuild data/thor_step2_fields.npz from the original sources.

You do not need to run this. The 1.8 MB result is committed, so the notebook
works straight after a clone. This script exists so the provenance of that
file is auditable and reproducible.

What it does:

1. Pulls wind stress curl, bathymetry and the six regime labels from the
   DNN4Cli repository (MIT licensed), verifying SHA-256.

2. Pulls the twenty yearly ECCO v4r3 sea-surface height files from the Google
   Drive folder linked in DNN4Cli (~580 MB), one at a time, folding each into
   a running mean and deleting it, so peak disk stays near 30 MB.

   The mean must be NaN-aware. THOR computes it as
   `monthlySSH['SSH'].mean(axis=0)`, and xarray skips NaN, so a cell that is
   NaN in all 240 months stays NaN. Summing with np.nansum and dividing by 240
   instead would turn land into a sea surface of exactly 0 m, which then puts a
   large false gradient in every ocean cell next to a coast -- and those cells
   survive the mask, because the mask is built from the gradients. This script
   divides by the per-cell count of finite months, which reproduces xarray's
   result exactly (verified: identical NaN mask, max abs difference 0.0).

3. Saves all four fields as float32 in one compressed npz. float32 costs about
   6e-8 m on SSH and the notebook casts to float32 before training anyway.

Latitude is deliberately not saved: the ECCO latitude field is exactly
linspace(-89.75, 89.75, 360) broadcast across longitude, to the last bit, so
the notebook reconstructs it.

Usage:  python make_dataset.py [--keep-ssh-raw]
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

import numpy as np

RAW = "https://raw.githubusercontent.com/maikejulie/DNN4Cli/main/THOR/Step2/Data_Step2"
SSH_FOLDER = "https://drive.google.com/drive/folders/1BgQn7Iog153MsXYD7QqblNSEJc9sRzE7"
OUT = Path("data/thor_step2_fields.npz")

# sha256 as retrieved on 2026-09-18
FILES = {
    "curlTau.npy":   "b4cb79fc5863aed51f27f4ddb7d1890c04ff1cb03ee1b9439e222118a6f5147c",
    "kCluster6.npy": "52a0181d7b7597b245c3961b05da69d137f6b804a4203da28c31c659f1750b80",
    "H_wHFacC.mat":  "f68fc1ddd27f2d7fe6329a3235783e7dd3ec70ee3d473113371606dd20bee4d0",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_small(tmp: Path) -> None:
    for name, expected in FILES.items():
        target = tmp / name
        if target.exists() and sha256(target) == expected:
            print(f"  {name:<16} cached, checksum ok")
            continue
        print(f"  {name:<16} downloading ...", end=" ", flush=True)
        urllib.request.urlretrieve(f"{RAW}/{name}", target)
        got = sha256(target)
        if got != expected:
            target.unlink(missing_ok=True)
            raise SystemExit(f"\nchecksum mismatch for {name}\n"
                             f"  expected {expected}\n  got      {got}")
        print("ok")


def ssh_mean(tmp: Path, keep_raw: bool) -> np.ndarray:
    import gdown
    import xarray as xr

    print(f"  SSH: listing {SSH_FOLDER}")
    # list first, then pull one at a time; download_folder() would fetch all
    # 580 MB before we could start averaging
    entries = gdown.download_folder(SSH_FOLDER, skip_download=True,
                                    quiet=True, use_cookies=False)
    entries = sorted((e for e in (entries or []) if str(e.path).endswith(".nc")),
                     key=lambda e: e.path)
    if not entries:
        raise SystemExit("the Drive folder returned no .nc files")

    raw = tmp / "ssh_raw"
    raw.mkdir(exist_ok=True)
    print(f"  SSH: {len(entries)} yearly files, averaging as they arrive")

    total = counts = None
    months = 0
    for entry in entries:
        path = raw / Path(entry.path).name
        if not path.exists():
            gdown.download(id=entry.id, output=str(path), quiet=True)
        with xr.open_dataset(path) as ds:
            block = ds["SSH"].values                     # (month, lat, lon)
            s = np.nansum(block, axis=0)
            c = np.isfinite(block).sum(axis=0)
            total = s if total is None else total + s
            counts = c if counts is None else counts + c
            months += block.shape[0]
        print(f"    {path.name}  {months:>3d} months", flush=True)
        if not keep_raw:
            path.unlink(missing_ok=True)

    if months != 240:
        print(f"  warning: expected 240 months, got {months}")
    # divide by finite months per cell, so all-NaN cells stay NaN
    return np.where(counts > 0, total / np.maximum(counts, 1), np.nan)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep-ssh-raw", action="store_true",
                    help="keep the 580 MB of yearly SSH files instead of deleting them")
    args = ap.parse_args()

    from scipy.io import loadmat

    tmp = Path("data/_sources")
    tmp.mkdir(parents=True, exist_ok=True)
    print("Rebuilding", OUT)
    fetch_small(tmp)

    # THOR transposes each field on read; keep that orientation
    curl  = np.transpose(np.load(tmp / "curlTau.npy"))
    label = np.transpose(np.load(tmp / "kCluster6.npy"))     # -1 over land
    bathy = np.transpose(loadmat(tmp / "H_wHFacC.mat")["val"])
    ssh   = ssh_mean(tmp, args.keep_ssh_raw)

    for name, a in (("curl", curl), ("bathy", bathy), ("label", label), ("ssh", ssh)):
        assert a.shape == (360, 720), f"{name} has shape {a.shape}"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT,
                        curl=curl.astype("float32"),
                        bathy=bathy.astype("float32"),
                        label=label.astype("float32"),
                        ssh=ssh.astype("float32"))
    print(f"\nwrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")
    print("  curl / bathymetry / labels: https://github.com/maikejulie/DNN4Cli (MIT)")
    print("  sea-surface height: ECCO v4 release 3, JPL/UT/MIT/AER")
    return 0


if __name__ == "__main__":
    sys.exit(main())
