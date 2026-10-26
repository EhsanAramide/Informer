"""Build and install both distribution forms outside the source checkout."""

from pathlib import Path
import os
import subprocess
import sys
import venv

import pytest


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run a distribution command with source-path injection disabled."""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    return subprocess.run(command, cwd=cwd, env=environment, text=True,
                          capture_output=True, timeout=180, check=False)


@pytest.fixture(scope="module")
def distributions(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Build one wheel and source distribution without network isolation."""
    output = tmp_path_factory.mktemp("distributions")
    completed = run(
        [sys.executable, "-m", "build", "--no-isolation", "--wheel", "--sdist",
         "--outdir", str(output), str(ROOT)], output,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    wheel = list(output.glob("informer-0.0.0-*.whl"))
    source = list(output.glob("informer-0.0.0.tar.gz"))
    assert len(wheel) == len(source) == 1
    return wheel[0], source[0]


@pytest.mark.parametrize("artifact_index", [0, 1], ids=["wheel", "sdist"])
def test_installed_artifact_imports_outside_checkout(
    tmp_path: Path, distributions: tuple[Path, Path], artifact_index: int
) -> None:
    """Install an artifact and smoke-test namespaced core imports externally."""
    environment = tmp_path / "environment"
    venv.EnvBuilder(with_pip=True, system_site_packages=True).create(environment)
    python = environment / "bin/python"
    artifact = distributions[artifact_index]
    completed = run(
        [str(python), "-m", "pip", "install", "--no-deps",
         "--no-build-isolation", str(artifact)], tmp_path,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    code = """
import pathlib
import sys
import informer
from informer import Informer, InformerStack
from informer.models.attn import ProbAttention
from informer.utils.masking import ProbMask
assert informer.__version__ == '0.0.0'
assert 'site-packages' in str(pathlib.Path(informer.__file__).resolve())
assert 'models' not in sys.modules and 'utils' not in sys.modules
assert Informer.__module__ == 'informer.models.model'
"""
    completed = run([str(python), "-I", "-B", "-c", code], tmp_path)
    assert completed.returncode == 0, completed.stdout + completed.stderr
