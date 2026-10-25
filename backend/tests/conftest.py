import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from floodguard.calibration import RunConfig  # noqa: E402
from floodguard.datasets.loaders import add_derived  # noqa: E402


def synthetic_frame(n=6000, seed=0, step=0.5):
    """Synthetic 30-min series with rain bursts followed by level rises (for unit tests only)."""
    rng = np.random.default_rng(seed)
    rain = np.where(rng.random(n) < 0.04, rng.gamma(1.5, 3.0, n), 0.0)
    kernel = np.exp(-np.arange(48) / 8.0)
    resp = np.convolve(rain, kernel)[:n] * 0.02
    level = np.round(0.8 + resp + rng.normal(0, 0.003, n), 2)
    ts = pd.date_range("2021-01-01", periods=n, freq="30min")
    df = pd.DataFrame({"t_hours": np.arange(n) * step, "timestamp": ts.astype(str), "level": level,
                       "rain__gauge": rain, "rain__radar": np.clip(rain * 0.5 + rng.normal(0, 0.05, n), 0, None)})
    return add_derived(df, step)


@pytest.fixture(scope="session")
def synth():
    return synthetic_frame()


@pytest.fixture(scope="session")
def cfg():
    return RunConfig(dataset="synthetic", step_hours=0.5, horizon_steps=2, rain_window_steps=4,
                     rain_sensors=("gauge", "radar"), level_bounds=(0.0, 10.0))


@pytest.fixture(scope="session")
def pipe(synth, cfg):
    from floodguard.calibration import split
    from floodguard.systems import Pipeline
    train, _ = split(synth, 0.7)
    return Pipeline.fit(train, cfg)
