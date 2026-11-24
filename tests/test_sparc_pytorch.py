"""Standalone tests for the SPARC PyTorch model (no TORAX dependency)."""

import subprocess
import sys

import numpy as np

from epednn_mit.models.sparc import pytorch_model


def test_sparc_pytorch_inference():
    model_out, test_out = pytorch_model.test_epednn_mit_sparc_pytorch()
    assert model_out.shape == test_out.shape
    np.testing.assert_allclose(model_out, test_out, rtol=1e-5, atol=1e-8)


def test_sparc_pytorch_cli():
    result = subprocess.run(
        [sys.executable, "-m", "epednn_mit.scripts.test", "sparc", "pytorch"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"CLI failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
