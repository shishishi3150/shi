"""Check launch-time CUDA configuration without requiring a local GPU."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("entry", ["train", "visualize", "evaluate", "export_policy", "quxian"])
def test_direct_script_imports_do_not_shadow_standard_library(entry):
    result = subprocess.run([sys.executable, f"scripts/{entry}.py", "--help"], cwd=ROOT,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("entry", ["script", "module", "main"])
@pytest.mark.parametrize("configured", [None, ":16:8"])
def test_cublas_config_is_ready_before_torch_import(entry, configured):
    env = os.environ.copy()
    env.pop("CUBLAS_WORKSPACE_CONFIG", None)
    if configured is not None:
        env["CUBLAS_WORKSPACE_CONFIG"] = configured
    expected = configured or ":4096:8"
    probe = f'''
import builtins, os, runpy, sys
original_import = builtins.__import__
def checked_import(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == {expected!r}, "cuBLAS configuration missing before torch import"
        print("CUBLAS_READY_BEFORE_TORCH")
        raise SystemExit(0)
    return original_import(name, *args, **kwargs)
builtins.__import__ = checked_import
if {entry!r} == "module":
    runpy.run_module("scripts.train", run_name="__main__")
else:
    runpy.run_path("scripts/train.py" if {entry!r} == "script" else "main.py", run_name="__main__")
raise AssertionError("entry did not import torch")
'''
    result = subprocess.run([sys.executable, "-B", "-c", probe], cwd=ROOT,
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CUBLAS_READY_BEFORE_TORCH" in result.stdout
