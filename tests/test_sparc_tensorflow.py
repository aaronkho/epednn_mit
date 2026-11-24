"""Standalone tests for the SPARC TensorFlow model (no TORAX dependency)."""

import subprocess
import sys

import numpy as np

from epednn_mit.models.sparc import tensorflow_model


def test_sparc_tensorflow_inference():
    model_out, test_out = tensorflow_model.test_epednn_mit_sparc_tensorflow()
    assert model_out.shape == test_out.shape
    np.testing.assert_allclose(model_out, test_out, rtol=1e-5, atol=1e-8)


def test_sparc_tensorflow_cli():
    result = subprocess.run(
        [sys.executable, "-m", "epednn_mit.scripts.test", "sparc", "tensorflow"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"CLI failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
