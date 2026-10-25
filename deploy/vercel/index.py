"""Vercel entrypoint: serves the FastAPI backend as a read-only hosted demo.

Bundled data (processed datasets and saved experiment summaries) lives in ./bundle;
the built React frontend is served statically from ./public.
"""
import os
from pathlib import Path

os.environ.setdefault("FLOODGUARD_ROOT", str(Path(__file__).resolve().parent / "bundle"))
os.environ.setdefault("FLOODGUARD_READONLY", "1")

from floodguard.api import app  # noqa: E402
