"""Responsive, read-only desktop scanner. Start with python -m src.app.main_gui."""
from __future__ import annotations

import argparse
import hashlib
import html
import sys
from pathlib import Path

import psutil
from PyQt6.QtCore import QObject, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (QApplication, QComboBox, QFileDialog, QLabel,
    QMainWindow, QPlainTextEdit, QProgressBar, QPushButton, QTabWidget,
    QTextBrowser, QVBoxLayout, QWidget)

from src.engine.offline_guidance import OfflineGuidance
from src.engine.decompiler import DisassemblyAnalyzer, MAX_BYTES
from src.engine.qnn_detector import DEFAULT_MODEL, HexaGuardDetector


class ScanWorker(QObject):
    """Own the inference session in a background thread for one scan."""
    success = pyqtSignal(object)
    failure = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, data: bytes | Path, architecture: str, options: dict):
        super().__init__()
        self.data, self.architecture, self.options = data, architecture, options

    def run(self):
        """Read a bounded input and emit plain data back to the UI."""
        try:
            if isinstance(self.data, Path):
                with self.data.open("rb") as stream:
                    data = stream.read(MAX_BYTES + 1)
            else:
                data = self.data
            if not data or len(data) > MAX_BYTES:
                raise ValueError("Select a nonempty file or script, no larger than 8 MiB")
            analysis = DisassemblyAnalyzer().analyze(data, self.architecture)
            prediction = None
            detector_options = {k: v for k, v in self.options.items() if k != "guidance_model"}
            if self.options.get("model_path"):
                prediction = HexaGuardDetector(**detector_options).predict(data)
            self.success.emit({"analysis": analysis, "prediction": prediction,
                "copilot": OfflineGuidance(self.options.get("guidance_model")).explain(prediction, analysis),
                "sha256": hashlib.sha256(data).hexdigest(), "hex": data[:4096].hex(" ")})
        except Exception as exc:
            self.failure.emit(f"{type(exc).__name__}: {exc}")
        finally:
            self.finished.emit()


class MainWindow(QMainWindow):
    """File drag-and-drop and script inspection with evidence-only rendering."""

    def __init__(self, options: dict):
        super().__init__()
        self.options, self.thread = options, None
        self.setWindowTitle("HexaGuard | Offline Endpoint Triage")
        self.resize(1050, 820)
        self.setAcceptDrops(True)
        panel = QWidget()
        layout = QVBoxLayout(panel)
        title = QLabel("HexaGuard — Offline Endpoint & Threat Disassembly")
        title.setStyleSheet("font-size:22px;font-weight:bold")
        layout.addWidget(title)
        self.status = QLabel("Ready — model inference" if options.get("model_path") else
                             "Static analysis mode — no trained model loaded")
        layout.addWidget(self.status)
        self.metrics = QLabel()
        layout.addWidget(self.metrics)
        self.score = QProgressBar()
        self.score.setRange(0, 100)
        self.score.setFormat("Threat probability: unavailable")
        layout.addWidget(self.score)
        self.input = QPlainTextEdit()
        self.input.setPlaceholderText("Paste a script, open a file, or drop a local file here")
        self.input.setMaximumHeight(160)
        layout.addWidget(self.input)
        self.arch = QComboBox()
        self.arch.addItems(["auto", "x64", "arm64"])
        layout.addWidget(QLabel("Architecture: auto reads PE headers; select ISA for raw code"))
        layout.addWidget(self.arch)
        self.scan = QPushButton("Analyze pasted text")
        self.scan.clicked.connect(lambda: self.start(self.input.toPlainText().encode("utf-8")))
        self.open = QPushButton("Open file")
        self.open.clicked.connect(self.choose_file)
        layout.addWidget(self.scan)
        layout.addWidget(self.open)
        tabs = QTabWidget()
        self.evidence, self.guide = QTextBrowser(), QTextBrowser()
        self.hex = QPlainTextEdit()
        self.hex.setReadOnly(True)
        for name, widget in [("Evidence / Disassembly", self.evidence),
                             ("MITRE ATT&CK / Guidance", self.guide), ("Hex (first 4 KiB)", self.hex)]:
            tabs.addTab(widget, name)
        layout.addWidget(tabs)
        self.setCentralWidget(panel)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_metrics)
        self.timer.start(1000)
        self.update_metrics()

    def update_metrics(self):
        """Display process RSS and host CPU utilization without blocking."""
        self.metrics.setText(f"Host CPU {psutil.cpu_percent():.0f}% | "
                            f"App RAM {psutil.Process().memory_info().rss / 2**20:.1f} MiB")

    def choose_file(self):
        """Ask for a local file; do not execute it."""
        path, _ = QFileDialog.getOpenFileName(self, "Analyze file")
        if path:
            self.start(Path(path))

    def dragEnterEvent(self, event):
        """Accept a single local file when idle."""
        urls = event.mimeData().urls()
        if self.thread is None and len(urls) == 1 and urls[0].isLocalFile():
            event.acceptProposedAction()

    def dropEvent(self, event):
        """Schedule the dropped file for background inspection."""
        self.start(Path(event.mimeData().urls()[0].toLocalFile()))

    def start(self, data: bytes | Path):
        """Prevent concurrent scans and clear stale findings."""
        if self.thread is not None:
            return
        self.status.setText("Analyzing locally…")
        self.score.setValue(0)
        self.score.setFormat("Threat probability: pending")
        self.evidence.clear()
        self.guide.clear()
        self.hex.clear()
        self.scan.setEnabled(False)
        self.open.setEnabled(False)
        self.thread = QThread(self)
        self.worker = ScanWorker(data, self.arch.currentText(), self.options)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.success.connect(self.show_result)
        self.worker.failure.connect(self.show_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.cleanup)
        self.thread.start()

    def cleanup(self):
        """Release the finished worker thread."""
        self.thread.deleteLater()
        self.thread = None
        self.scan.setEnabled(True)
        self.open.setEnabled(True)

    def show_error(self, message: str):
        """Present a failed scan without retaining a previous risk score."""
        self.status.setText("Analysis failed")
        self.score.setFormat("Threat probability: unavailable")
        self.evidence.setPlainText(message)

    def show_result(self, result: dict):
        """Escape every untrusted string before inserting rich text."""
        esc = lambda value: html.escape(str(value))
        pred, analysis = result["prediction"], result["analysis"]
        if pred:
            self.score.setValue(round(pred["threat_score"] * 100))
            self.score.setFormat(f"Model threat score: {pred['threat_score']:.1%}")
            if pred.get("is_demo"):
                self.score.setFormat(f"DEMO score (not a threat probability): {pred['threat_score']:.1%}")
            badge = ("Accelerated by Qualcomm Hexagon NPU" if pred["backend"] == "QNNExecutionProvider"
                     else "CPU inference")
            self.status.setText(f"{badge} | measured {pred['latency_ms']:.2f} ms | {pred['windows']} windows")
            if pred["fallback_reason"]:
                self.status.setToolTip(pred["fallback_reason"])
        else:
            self.score.setFormat("Threat probability: unavailable")
            self.status.setText("Static analysis complete — probability requires a trained model")
        lines = [f"<p>SHA-256: {esc(result['sha256'])}</p>",
                 f"<p>{esc(analysis['format'])} / {esc(analysis['architecture'])}</p>",
                 "<h3>Suspicious API indicators (not resolved calls)</h3>"]
        lines += [f"<p style='color:#c67800'>{esc(item)}</p>" for item in analysis["apis"]]
        lines += [f"<p>{esc(item)}</p>" for item in analysis["behaviors"] + analysis["warnings"]]
        lines.append("<pre>")
        for ins in analysis["instructions"]:
            line = esc(f"{ins['address']:>16}  {ins['mnemonic']:8} {ins['operands']}")
            lines.append((f"<b style='color:#b47700'>{line}</b>" if ins["is_call"] else line) + "\n")
        lines.append("</pre>")
        self.evidence.setHtml("".join(lines))
        guidance = result["copilot"]
        cards = [f"<p>{esc(guidance['summary'])}</p>"]
        for card in guidance["mappings"]:
            cards.append(f"<h3>{esc(card['id'])}: {esc(card['name'])}</h3>"
                         f"<p>{esc(card['tactic'])}<br>{esc(card['evidence'])}</p>")
        cards += [f"<p>• {esc(step)}</p>" for step in guidance["remediation"]]
        cards.append(f"<p>Guidance backend: {esc(guidance['reasoning_backend'])}</p>")
        if guidance.get("llm_guidance"):
            cards.append(f"<h3>Local model advice (unverified)</h3><p>{esc(guidance['llm_guidance'])}</p>")
        if guidance.get("fallback_reason"):
            cards.append(f"<p>{esc(guidance['fallback_reason'])}</p>")
        self.guide.setHtml("".join(cards))
        self.hex.setPlainText(result["hex"])

    def closeEvent(self, event):
        """Keep Qt alive while its worker still owns native inference resources."""
        if self.thread is not None:
            self.status.setText("Wait for the current analysis to finish before closing")
            event.ignore()
        else:
            event.accept()


def main():
    """Parse model selection and launch the desktop application."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", dest="model_path", default=str(DEFAULT_MODEL))
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument("--guidance-model", help="Local GGUF file or ORT GenAI model directory")
    parser.add_argument("--cpu-model", dest="cpu_model_path")
    parser.add_argument("--backend", choices=["auto", "cpu", "qnn"], default="auto")
    parser.add_argument("--qnn-backend", dest="backend_path", default="QnnHtp.dll")
    options = vars(parser.parse_args())
    if options.pop("static_only"):
        options["model_path"] = None
    app = QApplication(sys.argv[:1])
    window = MainWindow(options)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
