"""Byte/Hugging Face model inference with explicit whole-session fallback."""
from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter

import numpy as np

LOG = logging.getLogger(__name__)
DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "assets/models/threat_detector_int8.onnx"


class HexaGuardDetector:
    """Run fixed integer-token models with a float probability output [1, 1].

    Supports byte-v1 or the export's local HF tokenizer. The bundled synthetic
    model produces demo scores, never malicious/benign verdicts.
    """

    def __init__(self, model_path: str | None = None, backend: str = "auto", *,
                 cpu_model_path: str | None = None, backend_path: str = "QnnHtp.dll",
                 threshold: float = 0.7):
        """Attempt strict HTP inference; optionally fall back to a CPU model."""
        import onnxruntime as ort
        model_path = str(model_path or DEFAULT_MODEL)
        active_model_path = model_path
        if backend not in ("auto", "cpu", "qnn") or not 0 <= threshold <= 1:
            raise ValueError("Invalid backend or threshold")
        if not Path(model_path).is_file():
            raise FileNotFoundError(model_path)
        self.threshold = threshold
        self.backend = "CPUExecutionProvider"
        self.fallback_reason = None
        self.session = None
        if backend != "cpu":
            try:
                if "QNNExecutionProvider" not in ort.get_available_providers():
                    raise RuntimeError("Installed ONNX Runtime has no QNN provider")
                options = ort.SessionOptions()
                options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
                self.session = ort.InferenceSession(model_path, sess_options=options,
                    providers=[("QNNExecutionProvider", {"backend_path": backend_path,
                                "htp_performance_mode": "burst"})])
                if "QNNExecutionProvider" not in self.session.get_providers():
                    raise RuntimeError("ONNX Runtime did not activate QNN")
                self.session.disable_fallback()
                self.backend = "QNNExecutionProvider"
            except Exception as exc:
                self.session = None
                if backend == "qnn":
                    raise RuntimeError(f"Strict QNN initialization failed: {exc}") from exc
                self.fallback_reason = str(exc)
                LOG.warning("QNN unavailable; falling back to CPU: %s", exc)
        if self.session is None:
            cpu_path = cpu_model_path or model_path
            active_model_path = cpu_path
            self.session = ort.InferenceSession(cpu_path, providers=["CPUExecutionProvider"])
            self.session.disable_fallback()
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if not inputs or any(arg.type not in ("tensor(int32)", "tensor(int64)") for arg in inputs):
            raise ValueError("Model must accept integer token inputs")
        shape = inputs[0].shape
        if len(shape) != 2 or shape[0] != 1 or not isinstance(shape[1], int) or not 1 <= shape[1] <= 1048576:
            raise ValueError("Model input must have fixed shape [1, N], N <= 1048576")
        if len(outputs) != 1 or outputs[0].shape != [1, 1] or outputs[0].type != "tensor(float)":
            raise ValueError("Model output must be float32 probability [1, 1]")
        metadata = self.session.get_modelmeta().custom_metadata_map
        self.tokenizer_kind = metadata.get("hexaguard.tokenizer")
        self.model_status = metadata.get("hexaguard.model_status", "external-unvalidated")
        self.is_demo = self.model_status.endswith("demo")
        self.hf_tokenizer = None
        if self.tokenizer_kind == "byte-v1":
            if len(inputs) != 1:
                raise ValueError("byte-v1 requires one input")
        elif self.tokenizer_kind == "hf-local-v1":
            if {arg.name for arg in inputs} != {"input_ids", "attention_mask"}:
                raise ValueError("hf-local-v1 requires input_ids and attention_mask")
            if any(arg.shape != shape for arg in inputs):
                raise ValueError("All token inputs must have the same fixed shape")
            from transformers import AutoTokenizer
            self.hf_tokenizer = AutoTokenizer.from_pretrained(
                str(Path(active_model_path).resolve().parent / "tokenizer"),
                local_files_only=True, trust_remote_code=False)
        else:
            raise ValueError("Missing or unsupported model metadata: hexaguard.tokenizer")
        self.input_dtypes = {arg.name: np.int32 if arg.type == "tensor(int32)" else np.int64
                             for arg in inputs}
        self.input_name = inputs[0].name
        self.length = shape[1]
        self.dtype = np.int32 if inputs[0].type == "tensor(int32)" else np.int64

    def _feeds(self, data: bytes):
        """Yield complete windows using the export's tokenizer, with no network."""
        if self.hf_tokenizer is not None:
            from .decompiler import decode_text
            batch = self.hf_tokenizer(decode_text(data), max_length=self.length,
                padding="max_length", truncation=True, return_overflowing_tokens=True,
                return_tensors="np")
            if len(batch["input_ids"]) > 4096:
                raise ValueError("Input exceeds the 4096-window analysis budget")
            for i in range(len(batch["input_ids"])):
                yield {name: np.asarray(batch[name][i:i + 1], dtype=dtype)
                       for name, dtype in self.input_dtypes.items()}
        else:
            if (len(data) + self.length - 1) // self.length > 4096:
                raise ValueError("Input exceeds the 4096-window analysis budget")
            for offset in range(0, len(data), self.length):
                chunk = data[offset:offset + self.length]
                tokens = np.zeros((1, self.length), dtype=self.dtype)
                tokens[0, :len(chunk)] = np.frombuffer(chunk, dtype=np.uint8).astype(self.dtype) + 1
                yield {self.input_name: tokens}

    def predict(self, text_or_bytes: str | bytes) -> dict:
        """Classify nonoverlapping windows and return their maximum score.

        Latency includes tokenization and all windows; no content is executed.
        """
        from .decompiler import MAX_BYTES
        start = perf_counter()
        data = text_or_bytes.encode("utf-8") if isinstance(text_or_bytes, str) else text_or_bytes
        if not isinstance(data, bytes) or not data or len(data) > MAX_BYTES:
            raise ValueError("Input must contain 1 byte to 8 MiB")
        scores = []
        for feed in self._feeds(data):
            value = np.asarray(self.session.run(None, feed)[0])
            if value.shape != (1, 1) or not np.isfinite(value).all() or not 0 <= float(value[0, 0]) <= 1:
                raise ValueError("Model returned an invalid probability")
            scores.append(float(value[0, 0]))
        score = max(scores)
        return {"threat_score": score, "is_malicious": None if self.is_demo else score >= self.threshold,
                "latency_ms": (perf_counter() - start) * 1000, "backend": self.backend,
                "windows": len(scores), "fallback_reason": self.fallback_reason,
                "model_status": self.model_status, "is_demo": self.is_demo,
                "warning": "DEMO: untrained scores are not threat probabilities" if self.is_demo else
                           "External model requires independent threat-detection validation"}
