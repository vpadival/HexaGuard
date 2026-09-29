"""Export an optional Hugging Face classifier, or an explicit synthetic demo.

Downloads require --allow-download. Offline fallback needs only numpy, onnx and
onnxruntime. Random weights test the inference pipeline, not threat detection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.quantization import QuantType, quantize_dynamic

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "assets" / "models"


def synthetic_model(path: Path, length: int = 256) -> dict:
    """Construct a reproducible, input-dependent two-class demo graph."""
    rng = np.random.default_rng(20260929)
    tensors = {
        "scale": np.array(256.0, dtype=np.float32),
        "w1": rng.normal(0, 0.04, (length, 16)).astype(np.float32),
        "w2": rng.normal(0, 0.1, (16, 1)).astype(np.float32),
    }
    nodes = [helper.make_node("Cast", ["tokens"], ["floats"], to=TensorProto.FLOAT),
             helper.make_node("Div", ["floats", "scale"], ["normalized"]),
             helper.make_node("MatMul", ["normalized", "w1"], ["hidden"]),
             helper.make_node("Relu", ["hidden"], ["activation"]),
             helper.make_node("MatMul", ["activation", "w2"], ["logit"]),
             helper.make_node("Sigmoid", ["logit"], ["probability"])]
    graph = helper.make_graph(nodes, "hexaguard-synthetic-demo", [helper.make_tensor_value_info(
        "tokens", TensorProto.INT32, [1, length])], [helper.make_tensor_value_info(
        "probability", TensorProto.FLOAT, [1, 1])],
        [numpy_helper.from_array(value, name) for name, value in tensors.items()])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    helper.set_model_props(model, {"hexaguard.tokenizer": "byte-v1",
        "hexaguard.model_status": "synthetic-demo", "hexaguard.labels": "0=Benign,1=Malicious"})
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return {"status": "synthetic-demo", "source": "seeded random weights",
            "seed": 20260929, "tokenizer": "byte-v1", "sequence_length": length,
            "warning": "Untrained synthetic model; scores have no threat-detection meaning"}


def transformer_model(path: Path, source: str, allow_download: bool, length: int) -> dict:
    """Export class-1 softmax with the model's own locally saved tokenizer."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=not allow_download,
                                              trust_remote_code=False)
    model, loading = AutoModelForSequenceClassification.from_pretrained(
        source, num_labels=2, local_files_only=not allow_download,
        trust_remote_code=False, output_loading_info=True)
    model.eval()
    labels = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    if labels != {0: "benign", 1: "malicious"}:
        status = "unvalidated-demo"
    else:
        status = "external-unvalidated"
    if loading.get("missing_keys") or loading.get("mismatched_keys"):
        status = "untrained-head-demo"

    class ProbabilityModel(torch.nn.Module):
        def __init__(self, classifier):
            super().__init__()
            self.classifier = classifier

        def forward(self, input_ids, attention_mask):
            logits = self.classifier(input_ids=input_ids, attention_mask=attention_mask).logits
            return torch.softmax(logits, dim=-1)[:, 1:2]

    batch = tokenizer("HexaGuard harmless export sample", max_length=length,
                      padding="max_length", truncation=True, return_tensors="pt")
    with torch.no_grad():
        torch.onnx.export(ProbabilityModel(model), (batch["input_ids"], batch["attention_mask"]),
            str(path), input_names=["input_ids", "attention_mask"], output_names=["probability"],
            opset_version=17, dynamo=False)
    tokenizer.save_pretrained(path.parent / "tokenizer")
    artifact = onnx.load(path)
    helper.set_model_props(artifact, {"hexaguard.tokenizer": "hf-local-v1",
        "hexaguard.model_status": status, "hexaguard.labels": "0=Benign,1=Malicious"})
    onnx.checker.check_model(artifact)
    onnx.save(artifact, path)
    return {"status": status, "source": source, "tokenizer": "hf-local-v1",
            "sequence_length": length, "source_labels": labels,
            "warning": "Not validated for threat detection; a base model is not a fine-tuned classifier"}


def export_models(output: Path = DEFAULT_OUTPUT, *, source: str = "distilbert-base-uncased",
                  synthetic: bool = False, allow_download: bool = False, length: int = 256) -> dict:
    """Stage, quantize, validate both graphs and only then publish artifacts."""
    if not 8 <= length <= 512:
        raise ValueError("Sequence length must be between 8 and 512")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="export-", dir=output) as temporary:
        stage = Path(temporary)
        fp = stage / "threat_detector.onnx"
        int8 = stage / "threat_detector_int8.onnx"
        if synthetic:
            card = synthetic_model(fp, length)
        else:
            try:
                card = transformer_model(fp, source, allow_download, length)
            except Exception as exc:
                # Exception messages may contain authentication URLs; store the type only.
                card = synthetic_model(fp, length)
                card["fallback_reason"] = type(exc).__name__
                print(f"Transformer export unavailable ({type(exc).__name__}); using SYNTHETIC DEMO")
        quantize_dynamic(str(fp), str(int8), weight_type=QuantType.QInt8,
                         op_types_to_quantize=["MatMul", "Gemm"])
        import onnxruntime as ort
        for path in (fp, int8):
            onnx.checker.check_model(str(path))
            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            inputs = {arg.name: np.ones(arg.shape, dtype=np.int32 if arg.type == "tensor(int32)"
                                       else np.int64) for arg in session.get_inputs()}
            value = session.run(None, inputs)[0]
            if value.shape != (1, 1) or not np.isfinite(value).all() or not 0 <= value.item() <= 1:
                raise RuntimeError("Exported graph did not return a valid probability")
        quantized = onnx.load(int8)
        if not any(t.data_type == TensorProto.INT8 for t in quantized.graph.initializer):
            raise RuntimeError("Export contains no INT8 weights")
        card.update(labels={"0": "Benign", "1": "Malicious"},
                    quantization="ONNX Runtime dynamic INT8 weights; CPU artifact",
                    validated_for_threat_detection=False,
                    sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (fp, int8)})
        if card["tokenizer"] == "hf-local-v1":
            shutil.copytree(stage / "tokenizer", output / "tokenizer", dirs_exist_ok=True)
        for path in (fp, int8):
            path.replace(output / path.name)
        (output / "model_card.json").write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")
    return card


def main():
    """Offer offline, cache-only and explicitly connected export modes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source", default="distilbert-base-uncased")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--length", type=int, default=256)
    args = parser.parse_args()
    try:
        card = export_models(args.output, source=args.source, synthetic=args.synthetic,
                             allow_download=args.allow_download, length=args.length)
        print(json.dumps(card, indent=2))
    except Exception as exc:
        parser.exit(1, f"Export failed ({type(exc).__name__}): {exc}\n")


if __name__ == "__main__":
    main()
