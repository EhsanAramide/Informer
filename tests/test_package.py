"""Test the namespaced source package and frozen-reference parity."""

import importlib
from pathlib import Path
import subprocess
import sys

import pytest
import torch

import informer


ROOT = Path(__file__).resolve().parents[1]


def test_public_imports_and_version() -> None:
    """Expose root models and all preserved component/utility modules."""
    from informer import Informer, InformerStack
    from informer.models.model import Informer as ModuleInformer

    assert informer.__version__ == "0.0.0"
    assert Informer is ModuleInformer
    assert issubclass(Informer, torch.nn.Module)
    assert issubclass(InformerStack, torch.nn.Module)
    for module in ("attn", "decoder", "embed", "encoder", "model"):
        importlib.import_module(f"informer.models.{module}")
    for module in ("masking", "metrics", "tools"):
        importlib.import_module(f"informer.utils.{module}")
    assert "pandas" not in sys.modules


def test_time_module_has_actionable_missing_dependency_message(tmp_path: Path) -> None:
    """Keep pandas optional and explain the extra when its import is unavailable."""
    code = """
import importlib.abc
import sys
class BlockPandas(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'pandas' or fullname.startswith('pandas.'):
            raise ModuleNotFoundError("blocked", name='pandas')
sys.meta_path.insert(0, BlockPandas())
try:
    import informer.utils.timefeatures
except ModuleNotFoundError as error:
    assert "informer[time]" in str(error)
else:
    raise AssertionError("timestamp module unexpectedly imported")
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c",
         f"import sys; sys.path.insert(0, {str(ROOT / 'src')!r});" + code],
        cwd=tmp_path, text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("model", ["Informer", "InformerStack"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("attention", ["full", "prob"])
def test_reference_parity(
    tmp_path: Path, model: str, dtype: str, attention: str
) -> None:
    """Require exact package/reference predictions, attention, and gradients."""
    completed = subprocess.run(
        [sys.executable, "-I", "-B", str(ROOT / "tests/package_worker.py"),
         str(ROOT), model, dtype, attention],
        cwd=tmp_path, text=True, capture_output=True, timeout=90, check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("model", ["Informer", "InformerStack"])
@pytest.mark.parametrize(
    ("attention", "scenario"),
    [("full", "fixed_no_distil"), ("prob", "learned_distil"),
     ("full", "timef_no_distil")],
)
def test_reference_parity_customizations(
    tmp_path: Path, model: str, attention: str, scenario: str
) -> None:
    """Match reference embedding and distillation customization behavior."""
    completed = subprocess.run(
        [sys.executable, "-I", "-B", str(ROOT / "tests/package_worker.py"),
         str(ROOT), model, "float32", attention, scenario],
        cwd=tmp_path, text=True, capture_output=True, timeout=90, check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
