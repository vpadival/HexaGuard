"""Headless GUI integration tests with actual queued worker signals."""
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication
from src.app.main_gui import MainWindow
from src.engine.qnn_detector import DEFAULT_MODEL


def test_worker_and_error_state():
    app = QApplication.instance() or QApplication([])
    window = MainWindow({"model_path": None})
    window.start(b"# AmsiScanBuffer")
    deadline = time.monotonic() + 10
    while window.thread is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert window.thread is None
    assert "Static analysis complete" in window.status.text()
    assert "T1562.001" in window.guide.toPlainText()
    window.start(b"")
    deadline = time.monotonic() + 10
    while window.thread is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert window.thread is None
    assert window.status.text() == "Analysis failed"
    assert "nonempty" in window.evidence.toPlainText()
    window.close()


def test_bundled_model_is_labeled_demo():
    app = QApplication.instance() or QApplication([])
    window = MainWindow({"model_path": str(DEFAULT_MODEL), "backend": "cpu"})
    window.start(b"Write-Output 'hello'")
    deadline = time.monotonic() + 10
    while window.thread is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert window.thread is None
    assert "DEMO" in window.score.format()
    assert "synthetic/untrained" in window.guide.toPlainText()
    window.close()
