"""Standalone tests for the SPARC JAX model (no TORAX dependency)."""

import os
from pathlib import Path
import subprocess
import sys

import jax.numpy as jnp
import numpy as np
import pytest

from epednn_mit.models.sparc import jax_model


def test_sparc_jax_inference():
    model_out, test_out = jax_model.test_epednn_mit_sparc_jax()
    assert model_out.shape == test_out.shape
    assert model_out.dtype == jnp.float32
    np.testing.assert_allclose(model_out, test_out, rtol=1e-5, atol=1e-8)


def test_load_ensemble_params_empty_raises():
    with pytest.raises(ValueError, match="No parameters loaded"):
        jax_model.load_ensemble_params_from_pickle([])


def test_sparc_jax_cli_x32_and_x64():
    """Verify standalone JAX inference succeeds under both x32 and x64."""
    for x64 in ("0", "1"):
        env = {**os.environ, "JAX_ENABLE_X64": x64}
        result = subprocess.run(
            [sys.executable, "-m", "epednn_mit.scripts.test", "sparc", "jax"],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, (
            f"CLI failed with JAX_ENABLE_X64={x64}:\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def test_sparc_jax_single_and_ensemble_match_float32_under_x64():
    """Verify float64 inputs produce identical float32 outputs under x64."""
    script = """
from pathlib import Path
import jax.numpy as jnp
import numpy as np
from epednn_mit.models.sparc import jax_model

root = Path(jax_model.__file__).parent
files = sorted(root.glob("*sparc*.pkl"))
x64 = jnp.array([10.0, 9.0, 1.85, 0.57, 2.0, 0.45, 30.0, 1.2, 2.0], dtype=jnp.float64)
x32 = x64.astype(jnp.float32)

estats, eparams = jax_model.load_ensemble_params_from_pickle(files)
ens = jax_model.EPEDNNmitEnsemble()
out_e64 = ens.apply(eparams, x64, **estats)
out_e32 = ens.apply(eparams, x32, **estats)
assert out_e64.dtype == jnp.float32 and out_e32.dtype == jnp.float32
np.testing.assert_allclose(out_e64, out_e32)

sstats, sparams = jax_model.load_params_from_pickle(files[0])
single = jax_model.EPEDNNmit()
out_s64 = single.apply(sparams, x64, **sstats)
out_s32 = single.apply(sparams, x32, **sstats)
assert out_s64.dtype == jnp.float32 and out_s32.dtype == jnp.float32
np.testing.assert_allclose(out_s64, out_s32)
"""
    env = {**os.environ, "JAX_ENABLE_X64": "1"}
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"x64 precision check failed:\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
