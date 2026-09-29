"""Bundled model execution, artifact integrity, export and Hub regression tests."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import onnx
import pytest

from scripts import compile_ai_hub as hub_script
from scripts.export_threat_model import DEFAULT_OUTPUT, export_models
from src.engine.qnn_detector import DEFAULT_MODEL, HexaGuardDetector
from src.engine.offline_guidance import OfflineGuidance
from src.engine.decompiler import DisassemblyAnalyzer


def test_bundled_int8_cpu_fallback(monkeypatch):
    import onnxruntime as ort
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    detector = HexaGuardDetector()
    result = detector.predict("Write-Output 'hello'")
    assert result["backend"] == "CPUExecutionProvider"
    assert result["is_demo"] and result["is_malicious"] is None
    assert 0 <= result["threat_score"] <= 1
    assert result["fallback_reason"]
    graph = onnx.load(DEFAULT_MODEL)
    assert any(t.data_type == onnx.TensorProto.INT8 for t in graph.graph.initializer)
    card = json.loads((DEFAULT_OUTPUT / "model_card.json").read_text())
    for name, digest in card["sha256"].items():
        assert hashlib.sha256((DEFAULT_OUTPUT / name).read_bytes()).hexdigest() == digest


def test_offline_export_fallback(tmp_path, monkeypatch):
    from scripts import export_threat_model as exporter
    def unavailable(*args, **kwargs):
        raise OSError("offline")
    monkeypatch.setattr(exporter, "transformer_model", unavailable)
    card = export_models(tmp_path)
    assert card["status"] == "synthetic-demo"
    assert card["fallback_reason"] == "OSError"
    assert HexaGuardDetector(str(tmp_path / "threat_detector_int8.onnx"), backend="cpu").predict(b"test")["is_demo"]


def test_hub_mock_cli(tmp_path):
    env = dict(os.environ)
    env.pop("QAI_HUB_API_TOKEN", None)
    run = subprocess.run([sys.executable, "scripts/compile_ai_hub.py", "--dry-run", "--output", str(tmp_path)],
                         env=env, capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stderr
    assert "SIMULATED" in run.stdout
    report = json.loads((tmp_path / "profile_mock.json").read_text())
    assert report["simulated"] and report["compiled_binary"] is None
    assert report["npu"]["median_latency_ms"] == 4.2
    assert report["cpu"]["median_latency_ms"] == 38.6
    assert not list(tmp_path.glob("*.bin"))


def test_profile_units_and_missing_values():
    profile = {"execution_summary": {"all_inference_times": [4100, 4200, 4500],
        "estimated_inference_time": 4300, "inference_memory_peak_range": [100, 200]},
        "execution_detail": [{"compute_unit": "NPU"}]}
    result = hub_script.parse_profile(profile)
    assert result["median_latency_ms"] == 4.2
    assert result["estimated_latency_ms"] == 4.3
    assert result["peak_memory_bytes"] is None
    assert result["peak_memory_range_bytes"] == [100, 200]
    assert result["htp_only_verified"]
    assert hub_script.parse_profile({})["median_latency_ms"] is None
    assert not hub_script.parse_profile({})["htp_only_verified"]
    with pytest.raises(ValueError):
        hub_script.parse_profile({"execution_summary": {"all_inference_times": [float("nan")]}})


def test_authentication_does_not_persist(monkeypatch):
    calls = []
    fake = SimpleNamespace(ClientConfig=lambda **kw: kw,
                           Client=lambda **kw: calls.append(kw) or kw)
    monkeypatch.setitem(sys.modules, "qai_hub", fake)
    monkeypatch.setenv("QAI_HUB_API_TOKEN", "unit-test-not-a-secret")
    hub_script.authenticate()
    assert calls[0]["config"]["api_token"] == "unit-test-not-a-secret"


def test_live_orchestration_with_fake_sdk(tmp_path, monkeypatch):
    """Verify live call ordering/artifact types without pretending to use hardware."""
    calls = []
    class Artifact:
        def __init__(self, kind):
            self.kind = kind
        def download(self, path):
            assert self.kind == "context"
            Path(path).write_bytes(b"TEST-ONLY-CONTEXT")
    class Job:
        def __init__(self, kind):
            self.job_id = kind
            self.kind = kind
        def wait(self, timeout):
            return SimpleNamespace(success=True)
        def get_target_model(self):
            return Artifact(self.kind)
        def download_target_model(self, path):
            Path(path).write_bytes(b"TEST-ONLY-WRAPPER")
        def download_profile(self):
            return {"execution_summary": {"all_inference_times": [4200]},
                    "execution_detail": [{"compute_unit": "NPU"}]}
    class Client:
        def get_devices(self, name):
            assert name == hub_script.DEVICE
            return ["test-device"]
        def upload_model(self, path):
            calls.append(("upload", Path(path).name))
            return Artifact("uploaded")
        def submit_quantize_job(self, **kwargs):
            calls.append(("quantize", kwargs["weights_dtype"]))
            return Job("quantized")
        def submit_compile_job(self, **kwargs):
            calls.append(("compile", kwargs.get("options")))
            return Job("dlc" if "options" in kwargs else "wrapper")
        def submit_link_job(self, **kwargs):
            assert kwargs["models"].kind == "dlc"
            calls.append(("link", None))
            return Job("context")
        def submit_profile_job(self, **kwargs):
            calls.append(("profile", kwargs["options"]))
            return Job("profile")
    monkeypatch.setitem(sys.modules, "qai_hub", SimpleNamespace(QuantizeDtype=SimpleNamespace(INT8="int8")))
    monkeypatch.setattr(hub_script, "authenticate", lambda token: Client())
    calibration = tmp_path / "calibration.npz"
    np.savez(calibration, tokens=np.ones((2, 1, 256), dtype=np.int32))
    result = hub_script.compile_live(DEFAULT_MODEL, DEFAULT_OUTPUT / "threat_detector.onnx", calibration, tmp_path)
    assert not result["simulated"]
    assert ("compile", hub_script.COMPILE_OPTIONS) in calls
    assert ("profile", "--compute_unit npu") in calls
    assert ("profile", "--compute_unit cpu") in calls
    assert [step[0] for step in calls] == ["upload", "upload", "quantize", "compile", "link", "compile", "profile", "profile"]


def test_failed_job_is_an_error():
    job = SimpleNamespace(job_id="test-failed", wait=lambda timeout: SimpleNamespace(success=False))
    with pytest.raises(RuntimeError, match="failed"):
        hub_script.await_job(job)


def test_guidance_fallback_and_extra_mappings(tmp_path):
    analysis = DisassemblyAnalyzer().analyze("# NtUnmapViewOfSection SetThreadContext ResumeThread FromBase64String( IEX AmsiScanBuffer")
    result = OfflineGuidance(str(tmp_path / "absent.gguf")).explain(None, analysis)
    assert result["remediation"] and result["fallback_reason"]
    assert {"T1055.012", "T1059.001", "T1562.001"} <= {m["id"] for m in result["mappings"]}


def test_local_advice_keeps_rules(monkeypatch):
    analysis = DisassemblyAnalyzer().analyze("# AmsiScanBuffer")
    guidance = OfflineGuidance("local.gguf")
    monkeypatch.setattr(guidance, "_generate", lambda facts: "Preserve evidence and review endpoint logs.")
    result = guidance.explain(None, analysis)
    assert result["llm_guidance"] and result["mappings"] and result["remediation"]
    assert result["reasoning_backend"] == "local GGUF"
