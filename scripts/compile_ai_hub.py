"""Explicitly online model preparation; never invoked by the offline app."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    """Quantize with calibration or compile a pre-quantized w4a16 artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--device", default="Snapdragon X Elite CRD")
    parser.add_argument("--quantization", choices=["w8a8", "w4a16"], default="w8a8")
    parser.add_argument("--prequantized", action="store_true")
    parser.add_argument("--calibration", type=Path, help="NPZ input-name arrays shaped [samples,1,N]")
    parser.add_argument("--output", type=Path, default=Path("artifacts/ai-hub"))
    parser.add_argument("--allow-upload", action="store_true", help="Authorize uploading model and calibration")
    args = parser.parse_args()
    if not args.allow_upload:
        parser.error("AI Hub is online. Pass --allow-upload to upload the model and calibration data")
    if not args.model.exists():
        parser.error("Model path does not exist")
    if args.quantization == "w4a16" and not args.prequantized:
        parser.error("w4a16 requires a compatible externally quantized artifact; no generic conversion is assumed")
    if not args.prequantized and args.calibration is None:
        parser.error("w8a8 quantization requires representative --calibration data")
    import numpy as np
    import qai_hub as hub

    args.output.mkdir(parents=True, exist_ok=True)
    try:
        model = str(args.model)
        jobs = []
        if not args.prequantized:
            with np.load(args.calibration, allow_pickle=False) as archive:
                calibration = {}
                for name in archive.files:
                    values = archive[name]
                    if values.ndim != 3 or values.shape[0] == 0 or values.shape[1] != 1:
                        raise ValueError(f"{name}: expected [samples,1,N]")
                    calibration[name] = [values[i] for i in range(len(values))]
            if not calibration:
                raise ValueError("Calibration archive is empty")
            quant = hub.submit_quantize_job(model=model, calibration_data=calibration,
                weights_dtype=hub.QuantizeDtype.INT8, activations_dtype=hub.QuantizeDtype.INT8)
            model = quant.get_target_model()
            if model is None:
                raise RuntimeError("Quantization failed; inspect AI Hub job details")
            jobs.append(str(quant.job_id))
        device = hub.Device(args.device)
        _, link = hub.submit_compile_and_link_jobs(models=model, device=device)
        context = link.get_target_model()
        if context is None:
            raise RuntimeError("QNN compilation/link failed; inspect AI Hub job details")
        context.download(str(args.output / "hexaguard.bin"))
        wrapped = hub.submit_compile_job(model=context, device=device)
        if wrapped.get_target_model() is None:
            raise RuntimeError("ONNX wrapping failed; inspect AI Hub job details")
        wrapped.download_target_model(str(args.output / "hexaguard.onnx"))
        jobs.extend([str(link.job_id), str(wrapped.job_id)])
        (args.output / "build.json").write_text(json.dumps({"device": args.device,
            "quantization_requested": args.quantization, "prequantized": args.prequantized,
            "jobs": jobs, "note": "Validate actual tensor contract, encodings and target accuracy"}, indent=2), encoding="utf-8")
        print(f"Downloaded artifacts to {args.output.resolve()}")
    except Exception as exc:
        parser.exit(1, f"AI Hub build failed: {exc}\n")


if __name__ == "__main__":
    main()
