"""Strict byte-token model interface with explicit whole-session fallback."""
from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter

import numpy as np

LOG = logging.getLogger(__name__)


class HexaGuardDetector:
    """Run a supplied model trained for byte-v1: padding=0, byte IDs=1..256.

    Contract: one int32/int64 input [1, N], N fixed; one float probability
    output [1, 1]. Scores are only as reliable as the supplied model.
    """

    def __init__(self, model_path: str, backend: str = "auto", *,
                 cpu_model_path: str | None = None, backend_path: str = "QnnHtp.dll",
                 threshold: float = 0.7):
        """Attempt strict HTP inference; optionally fall back to a CPU model."""
        import onnxruntime as ort
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
            self.session = ort.InferenceSession(cpu_path, providers=["CPUExecutionProvider"])
            self.session.disable_fallback()
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or inputs[0].type not in ("tensor(int32)", "tensor(int64)"):
            raise ValueError("Model must accept one integer byte-v1 input")
        shape = inputs[0].shape
        if len(shape) != 2 or shape[0] != 1 or not isinstance(shape[1], int) or not 1 <= shape[1] <= 1048576:
            raise ValueError("Model input must have fixed shape [1, N], N <= 1048576")
        if len(outputs) != 1 or outputs[0].shape != [1, 1] or outputs[0].type != "tensor(float)":
            raise ValueError("Model output must be float32 probability [1, 1]")
        metadata = self.session.get_modelmeta().custom_metadata_map
        if metadata.get("hexaguard.tokenizer") != "byte-v1":
            raise ValueError("Missing model metadata: hexaguard.tokenizer=byte-v1")
        self.input_name = inputs[0].name
        self.length = shape[1]
        self.dtype = np.int32 if inputs[0].type == "tensor(int32)" else np.int64

    def predict(self, text_or_bytes: str | bytes) -> dict:
        """Classify nonoverlapping windows and return their maximum score.

        Latency includes tokenization and all windows; no content is executed.
        """
        from .decompiler import MAX_BYTES
        start = perf_counter()
        data = text_or_bytes.encode("utf-8") if isinstance(text_or_bytes, str) else text_or_bytes
        if not isinstance(data, bytes) or not data or len(data) > MAX_BYTES:
            raise ValueError("Input must contain 1 byte to 8 MiB")
        count = (len(data) + self.length - 1) // self.length
        if count > 4096:
            raise ValueError("Input exceeds the 4096-window analysis budget")
        scores = []
        for offset in range(0, len(data), self.length):
            chunk = data[offset:offset + self.length]
            tokens = np.zeros((1, self.length), dtype=self.dtype)
            tokens[0, :len(chunk)] = np.frombuffer(chunk, dtype=np.uint8).astype(self.dtype) + 1
            value = np.asarray(self.session.run(None, {self.input_name: tokens})[0])
            if value.shape != (1, 1) or not np.isfinite(value).all() or not 0 <= float(value[0, 0]) <= 1:
                raise ValueError("Model returned an invalid probability")
            scores.append(float(value[0, 0]))
        score = max(scores)
        return {"threat_score": score, "is_malicious": score >= self.threshold,
                "latency_ms": (perf_counter() - start) * 1000, "backend": self.backend,
                "windows": count, "fallback_reason": self.fallback_reason}
