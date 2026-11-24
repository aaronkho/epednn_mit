"""Standalone tests for epednn_mit utility functions (no TORAX dependency)."""

from pathlib import Path

from epednn_mit.models import sparc
from epednn_mit.utils.load import load_weights


def test_load_weights_sparc():
    sparc_dir = Path(sparc.__file__).parent
    weights_files = sorted(sparc_dir.glob("*sparc*.pkl"))
    assert len(weights_files) == 10

    weights_list = load_weights(weights_files)
    assert len(weights_list) == 10
    expected_keys = {
        "input.mean",
        "input.variance",
        "output.mean",
        "output.variance",
        "hidden_layer0.weight",
        "hidden_layer0.bias",
        "hidden_layer1.weight",
        "hidden_layer1.bias",
        "output_layer.weight",
        "output_layer.bias",
    }
    for weights in weights_list:
        assert expected_keys.issubset(weights.keys())


def test_load_weights_skips_missing_files(tmp_path):
    missing = tmp_path / "does_not_exist.pkl"
    assert load_weights([missing]) == []
