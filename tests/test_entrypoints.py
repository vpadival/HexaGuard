"""Regression checks for direct-file commands from any working directory."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_gui_direct_file_help(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "src/app/main_gui.py"), "--help"],
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "--model" in result.stdout


def test_benchmark_without_arguments(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "scripts/run_benchmark.py")],
                            cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "100 runs" in result.stdout
    assert "Not measured" in result.stdout
    assert "Not run" in result.stdout
    assert "Unavailable" not in result.stdout
