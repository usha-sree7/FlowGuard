"""Assemble a read-only Vercel deployment of HydroSentinel AI in deploy/build/.

Usage (from the repo root, after importing the datasets and building the frontend):
    python deploy/build_vercel.py
    cd deploy/build && vercel deploy --prod

The build folder holds the backend package, the processed datasets, the saved experiment
summaries (results.json + summary.csv only) and the built frontend (served from public/).
Imports and new experiments are disabled on the hosted copy (FLOODGUARD_READONLY=1).
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = Path(__file__).resolve().parent / "vercel"
BUILD = Path(__file__).resolve().parent / "build"


def main() -> None:
    dist = ROOT / "frontend" / "dist"
    processed = ROOT / "data" / "processed"
    if not (dist / "index.html").exists():
        raise SystemExit("frontend/dist missing: run `npm run build` in frontend/ first")
    if not any(processed.glob("*.csv")):
        raise SystemExit("data/processed is empty: import the datasets first")

    link = BUILD / ".vercel"
    saved_link = None
    if link.exists():  # keep the project link across rebuilds
        saved_link = Path(shutil.copytree(link, BUILD.parent / ".vercel-link", dirs_exist_ok=True))
    shutil.rmtree(BUILD, ignore_errors=True)
    BUILD.mkdir(parents=True, exist_ok=True)  # Windows can leave the emptied folder behind briefly

    shutil.copytree(TEMPLATE, BUILD, dirs_exist_ok=True)
    shutil.copytree(ROOT / "backend" / "floodguard", BUILD / "floodguard",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(processed, BUILD / "bundle" / "data" / "processed")
    for run in sorted((ROOT / "results" / "runs").glob("*")):
        out = BUILD / "bundle" / "results" / "runs" / run.name
        out.mkdir(parents=True)
        for name in ("results.json", "summary.csv"):
            if (run / name).exists():
                shutil.copy2(run / name, out / name)
    for f in (ROOT / "results").glob("comparison_*.json"):
        shutil.copy2(f, BUILD / "bundle" / "results" / f.name)
    shutil.copytree(dist, BUILD / "public")

    if saved_link:
        shutil.copytree(saved_link, link)
        shutil.rmtree(saved_link)
    print(f"built {BUILD}\nnext: cd {BUILD} && vercel deploy --prod")


if __name__ == "__main__":
    main()
