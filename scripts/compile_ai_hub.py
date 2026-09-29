"""Official AI Hub build/profile workflow with explicitly simulated demos.

Dynamic INT8 is the CPU baseline. HTP receives calibrated static INT8 through
QNN DLC compilation and linking, not the obsolete Windows library target.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "assets/models"
DEVICE = "Snapdragon X Elite CRD"
COMPILE_OPTIONS = "--target_runtime qnn_dlc --compute_unit npu"


def authenticate(api_token: str | None = None):
    """Create an SDK client without persisting credentials."""
    token = api_token or os.environ.get("QAI_HUB_API_TOKEN")
    if not token or not token.strip():
        raise ValueError("Set QAI_HUB_API_TOKEN or pass --api-token for live mode")
    import qai_hub as hub
    return hub.Client(config=hub.ClientConfig(api_token=token.strip(), verbose=False))


def target_device(client):
    """Verify the requested device is available to the account."""
    devices = client.get_devices(name=DEVICE)
    if not devices:
        raise RuntimeError(f"Device unavailable: {DEVICE}")
    return devices[0]


def upload_model(client, path: Path):
    """Upload a caller-selected ONNX artifact."""
    if not path.is_file():
        raise FileNotFoundError(path)
    return client.upload_model(str(path))


def await_job(job, timeout: int = 1800):
    """Bound job waits; service failure never becomes mock success."""
    if not job.wait(timeout=timeout).success:
        raise RuntimeError(f"AI Hub job {job.job_id} failed or timed out; inspect its Hub page")


def submit_compile(client, model, device):
    """Compile static INT8 to DLC, then link a genuine QNN context binary."""
    compile_job = client.submit_compile_job(model=model, device=device, options=COMPILE_OPTIONS)
    await_job(compile_job)
    dlc = compile_job.get_target_model()
    if dlc is None:
        raise RuntimeError("Compile job returned no target model")
    link_job = client.submit_link_job(models=dlc, device=device)
    await_job(link_job)
    context = link_job.get_target_model()
    if context is None:
        raise RuntimeError("Link job returned no context binary")
    return context, [compile_job.job_id, link_job.job_id]


def parse_profile(profile: dict) -> dict:
    """Convert SDK microseconds/bytes without inventing missing measurements."""
    summary = profile.get("execution_summary", {})

    def valid(value):
        return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0

    samples = summary.get("all_inference_times") or []
    if not all(valid(v) for v in samples):
        raise ValueError("Invalid profile time samples")
    estimated = summary.get("estimated_inference_time")
    memory = summary.get("estimated_inference_peak_memory")
    peak_range = summary.get("inference_memory_peak_range")
    units = sorted({row.get("compute_unit", "UNKNOWN") for row in profile.get("execution_detail", [])})
    return {"median_latency_ms": statistics.median(samples) / 1000 if samples else None,
            "estimated_latency_ms": estimated / 1000 if valid(estimated) else None,
            "peak_memory_bytes": memory if valid(memory) else None,
            "peak_memory_range_bytes": list(peak_range) if peak_range and all(valid(v) for v in peak_range) else None,
            "compute_units": units, "htp_only_verified": units == ["NPU"]}


def profile_model(client, model, device, compute_unit: str):
    """Return normalized and raw hardware measurements plus job identity."""
    job = client.submit_profile_job(model=model, device=device, options=f"--compute_unit {compute_unit}")
    await_job(job)
    raw = job.download_profile()
    if not isinstance(raw, dict):
        raise ValueError("AI Hub returned an invalid profile")
    return parse_profile(raw), raw, job.job_id


def load_calibration(path: Path, fp_model: Path) -> dict:
    """Validate representative inputs locally before uploading anything."""
    import numpy as np
    import onnxruntime as ort
    session = ort.InferenceSession(str(fp_model), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {arg.name for arg in inputs}:
            raise ValueError("Calibration keys must match model inputs")
        calibration, counts = {}, set()
        for arg in inputs:
            values = archive[arg.name]
            dtype = {"tensor(int32)": np.dtype("int32"), "tensor(int64)": np.dtype("int64")}.get(arg.type)
            if values.ndim < 1 or dtype is None or values.dtype != dtype or list(values.shape[1:]) != arg.shape or not len(values):
                raise ValueError(f"Invalid calibration shape/dtype for {arg.name}")
            counts.add(len(values))
            calibration[arg.name] = [values[i] for i in range(len(values))]
        if len(counts) != 1:
            raise ValueError("Calibration inputs need identical sample counts")
    return calibration


def compile_live(model_path: Path, fp_model: Path, calibration_path: Path, output: Path,
                 api_token: str | None = None) -> dict:
    """Upload INT8 baseline, calibrate float source, compile, download and profile."""
    import qai_hub as hub
    calibration = load_calibration(calibration_path, fp_model)
    client = authenticate(api_token)
    device = target_device(client)
    cpu_model = upload_model(client, model_path)
    float_model = upload_model(client, fp_model)
    quantize_job = client.submit_quantize_job(model=float_model, calibration_data=calibration,
        weights_dtype=hub.QuantizeDtype.INT8, activations_dtype=hub.QuantizeDtype.INT8)
    await_job(quantize_job)
    quantized = quantize_job.get_target_model()
    if quantized is None:
        raise RuntimeError("Quantization job returned no model")
    context, job_ids = submit_compile(client, quantized, device)
    output.mkdir(parents=True, exist_ok=True)
    context_path = output / "threat_detector_qnn.bin"
    context.download(str(context_path))
    if not context_path.is_file() or not context_path.stat().st_size:
        raise RuntimeError("Context download is missing or empty")
    wrapper = client.submit_compile_job(model=context, device=device)
    await_job(wrapper)
    wrapper.download_target_model(str(output / "threat_detector_qnn.onnx"))
    npu, raw_npu, npu_id = profile_model(client, context, device, "npu")
    cpu, raw_cpu, cpu_id = profile_model(client, cpu_model, device, "cpu")
    result = {"mode": "hardware", "simulated": False, "device": DEVICE,
        "compile_options": COMPILE_OPTIONS, "npu": npu, "cpu": cpu,
        "compiled_binary": str(context_path.resolve()),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "jobs": [quantize_job.job_id, *job_ids, wrapper.job_id, npu_id, cpu_id]}
    for name, report in [("npu_raw", raw_npu), ("cpu_raw", raw_cpu), ("hardware", result)]:
        (output / f"profile_{name}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return result


def mock_profile(model_path: Path, output: Path) -> dict:
    """Load a labeled fixture, with no network and no manufactured .bin."""
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    result = json.loads((ROOT / "tests/fixtures/ai_hub_mock_profile.json").read_text(encoding="utf-8"))
    result["model_sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
    result["compile_options"] = COMPILE_OPTIONS
    output.mkdir(parents=True, exist_ok=True)
    (output / "profile_mock.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    """No token defaults to mock; --live fails if credentials are missing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, nargs="?", default=MODEL_DIR / "threat_detector_int8.onnx")
    parser.add_argument("--float-model", type=Path, default=MODEL_DIR / "threat_detector.onnx")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--output", type=Path, default=MODEL_DIR)
    parser.add_argument("--api-token", help="Prefer QAI_HUB_API_TOKEN over shell history exposure")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", "--mock", dest="dry_run", action="store_true")
    mode.add_argument("--live", action="store_true")
    parser.add_argument("--allow-upload", action="store_true")
    args = parser.parse_args()
    token = args.api_token or os.environ.get("QAI_HUB_API_TOKEN")
    mock = args.dry_run or (not token and not args.live)
    if not mock and not args.allow_upload:
        parser.error("Live mode uploads models/calibration. Pass --allow-upload")
    if not mock and not args.calibration:
        parser.error("HTP static INT8 requires representative --calibration NPZ")
    try:
        if mock:
            print("SIMULATED DEMO METRICS: no hardware job ran; no QNN binary was generated")
            result = mock_profile(args.model, args.output)
        else:
            result = compile_live(args.model, args.float_model, args.calibration, args.output, token)
        print(json.dumps(result, indent=2))
    except Exception as exc:
        message = str(exc).replace(token, "[REDACTED]") if token else str(exc)
        parser.exit(1, f"AI Hub build failed ({type(exc).__name__}): {message}\n")


if __name__ == "__main__":
    main()
