"""Defensive fixtures and inference contract regression tests."""
from pathlib import Path

import numpy as np
import pytest

from src.engine.copilot import ThreatCopilot
from src.engine.decompiler import DisassemblyAnalyzer, MAX_BYTES
from src.engine.qnn_detector import HexaGuardDetector

FIXTURES = Path(__file__).parent


def test_benign_fixture():
    analysis = DisassemblyAnalyzer().analyze((FIXTURES / "sample_benign.ps1").read_bytes())
    assert not analysis["apis"]
    assert not analysis["behaviors"]
    assert not ThreatCopilot().explain(None, analysis)["mappings"]


def test_inert_indicators():
    analysis = DisassemblyAnalyzer().analyze((FIXTURES / "sample_malicious.ps1").read_bytes())
    assert {"base64_wrapper", "amsi_reference", "string_concatenation", "variable_mutation"} <= set(analysis["behaviors"])
    result = ThreatCopilot().explain(None, analysis)
    assert {card["id"] for card in result["mappings"]} == {"T1055", "T1562.001", "T1027"}
    assert "unavailable" in result["summary"]


def test_utf16():
    analysis = DisassemblyAnalyzer().analyze("# AmsiScanBuffer".encode("utf-16"))
    assert "amsi_reference" in analysis["behaviors"]


@pytest.mark.parametrize("arch,data,mnemonic", [("x64", b"\x90\xc3", "nop"),
    ("arm64", bytes.fromhex("1f2003d5c0035fd6"), "nop")])
def test_raw_disassembly(arch, data, mnemonic):
    analysis = DisassemblyAnalyzer().analyze(data, arch)
    assert analysis["instructions"][0]["mnemonic"] == mnemonic


def test_malformed_and_limits():
    assert DisassemblyAnalyzer().analyze(b"MZbad")["warnings"]
    with pytest.raises(ValueError):
        DisassemblyAnalyzer().analyze(b"x" * (MAX_BYTES + 1))
    assert not DisassemblyAnalyzer().analyze(b"\x90\xc3")["instructions"]


@pytest.fixture
def model(tmp_path):
    """Build a constant model ONLY to exercise integration, not detection."""
    import onnx
    from onnx import TensorProto, helper
    node = helper.make_node("Constant", [], ["probability"], value=helper.make_tensor(
        "value", TensorProto.FLOAT, [1, 1], [0.25]))
    graph = helper.make_graph([node], "test-only", [helper.make_tensor_value_info(
        "tokens", TensorProto.INT32, [1, 8])], [helper.make_tensor_value_info(
        "probability", TensorProto.FLOAT, [1, 1])])
    artifact = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    artifact.ir_version = 9
    helper.set_model_props(artifact, {"hexaguard.tokenizer": "byte-v1"})
    path = tmp_path / "contract.onnx"
    onnx.save(artifact, path)
    return path


def test_real_cpu_inference(model):
    detector = HexaGuardDetector(str(model), backend="cpu")
    result = detector.predict("123456789")
    assert result["threat_score"] == 0.25
    assert result["windows"] == 2
    assert not result["is_malicious"]
    assert result["backend"] == "CPUExecutionProvider"
    assert result["latency_ms"] >= 0
    with pytest.raises(ValueError):
        detector.predict(b"")


def test_cpu_fallback(model, monkeypatch, caplog):
    import onnxruntime as ort
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    result = HexaGuardDetector(str(model)).predict("fixture")
    assert result["backend"] == "CPUExecutionProvider"
    assert "falling back to CPU" in caplog.text
    with pytest.raises(RuntimeError, match="Strict QNN"):
        HexaGuardDetector(str(model), backend="qnn")


def test_bad_metadata(model):
    import onnx
    artifact = onnx.load(model)
    artifact.ClearField("metadata_props")
    onnx.save(artifact, model)
    with pytest.raises(ValueError, match="metadata"):
        HexaGuardDetector(str(model), backend="cpu")


def test_byte_tokenization_and_invalid_output(model):
    detector = HexaGuardDetector(str(model), backend="cpu")

    class RecordingSession:
        def run(self, _, inputs):
            assert inputs["tokens"].tolist() == [[1, 256, 0, 0, 0, 0, 0, 0]]
            return [np.array([[float("nan")]], dtype=np.float32)]

    detector.session = RecordingSession()
    with pytest.raises(ValueError, match="invalid probability"):
        detector.predict(b"\x00\xff")


def test_benchmark_console(model):
    """Exercise 100-run reporting on a legacy Windows console encoding."""
    import os
    import subprocess
    import sys
    environment = dict(os.environ, PYTHONIOENCODING="cp1252")
    result = subprocess.run([sys.executable, "scripts/run_benchmark.py",
        "--cpu-model", str(model), "--qnn-model", str(model),
        "--input", str(FIXTURES / "sample_benign.ps1")],
        capture_output=True, text=True, env=environment, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "100 runs" in result.stdout
    assert "Not measured" in result.stdout
