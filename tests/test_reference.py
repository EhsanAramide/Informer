"""Characterize pinned upstream sources without importing their generic modules."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_reference_source_integrity() -> None:
    """Require byte-identical reference files, independent of Git availability."""
    manifest = json.loads((ROOT / "tests/reference_manifest.json").read_text())
    assert manifest["revision"] == "29f2a739226a509202a092b464163da81fa74960"
    for relative, expected in manifest["git_blobs"].items():
        contents = (ROOT / relative).read_bytes()
        blob = f"blob {len(contents)}\0".encode() + contents
        assert hashlib.sha1(blob).hexdigest() == expected, relative


def _run_reference(tmp_path: Path, *arguments: str) -> dict:
    """Run a bytecode-free isolated interpreter outside the reference checkout.

    Args:
        tmp_path: Pytest-owned working directory for process isolation.
        arguments: Worker case/model/dtype/attention selections.

    Returns:
        Worker JSON evidence after successful numerical assertions.
    """
    before = {name for name in sys.modules if name == "models" or name.startswith("models.")
              or name == "utils" or name.startswith("utils.")}
    completed = subprocess.run(
        [sys.executable, "-I", "-B", str(ROOT / "tests/reference_worker.py"),
         str(ROOT), *arguments], cwd=tmp_path, text=True, capture_output=True,
        timeout=90, check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    # Keep dependency warnings visible rather than suppressing child stderr.
    if completed.stderr:
        print(completed.stderr, file=sys.stderr)
    after = {name for name in sys.modules if name == "models" or name.startswith("models.")
             or name == "utils" or name.startswith("utils.")}
    assert before == after
    return json.loads(completed.stdout)


@pytest.mark.parametrize("model", ["Informer", "InformerStack"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("attention", ["full", "prob"])
def test_reference_model_contract(tmp_path: Path, model: str, dtype: str, attention: str) -> None:
    """Check unequal widths, gradients, returns, RNG replay and checkpoint parity."""
    report = _run_reference(tmp_path, "model", model, dtype, attention)
    assert report["prediction_shape"] == [2, 4, 3]
    assert report["model"] == model and report["dtype"] == dtype
    assert report["checks_passed"] is True


def test_reference_behavior_and_limitations(tmp_path: Path) -> None:
    """Record stochastic evaluation, distillation, circular embedding and limits."""
    report = _run_reference(tmp_path, "behavior")
    assert report["checks_passed"] is True
