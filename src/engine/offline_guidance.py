"""Optional local GGUF/ORT GenAI guidance with unconditional rule fallback."""
from __future__ import annotations

import json
from pathlib import Path

from .copilot import ThreatCopilot


class OfflineGuidance:
    """Retain rule evidence and add bounded, advisory-only local generation."""

    def __init__(self, model_path: str | None = None, *, max_tokens: int = 256):
        self.model_path = Path(model_path) if model_path else None
        self.max_tokens = max(1, min(max_tokens, 512))
        self.runner = None

    def _generate(self, facts: dict) -> str:
        """Load only local weights. No tools, raw samples, or remote APIs."""
        prompt = ("You are a defensive endpoint triage assistant. Explain these static "
                  "indicators with uncertainty. Recommend evidence preservation and manual review. "
                  "Do not provide executable commands or infer that compromise is proven.\n" +
                  json.dumps(facts, ensure_ascii=True))
        path = self.model_path
        if path is None or not path.exists():
            raise FileNotFoundError("Local reasoning weights are not configured or missing")
        if path.suffix.lower() == ".gguf":
            from llama_cpp import Llama
            if self.runner is None:
                self.runner = Llama(model_path=str(path), n_ctx=2048, n_gpu_layers=0, verbose=False)
            result = self.runner.create_chat_completion(messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens, temperature=0.0)
            return result["choices"][0]["message"]["content"]
        if path.is_dir() and (path / "genai_config.json").is_file():
            import onnxruntime_genai as og
            if self.runner is None:
                self.runner = og.Model(str(path))
            tokenizer = og.Tokenizer(self.runner)
            # An explicit local template preserves model-specific chat formatting.
            template = (path / "chat_template.txt").read_text(encoding="utf-8")
            if "{prompt}" not in template:
                raise ValueError("chat_template.txt must contain {prompt}")
            tokens = tokenizer.encode(template.replace("{prompt}", prompt))
            if len(tokens) > 1536:
                raise ValueError("Guidance prompt exceeds the context budget")
            params = og.GeneratorParams(self.runner)
            params.set_search_options(max_length=len(tokens) + self.max_tokens, do_sample=False)
            generator = og.Generator(self.runner, params)
            try:
                generator.append_tokens(tokens)
                for _ in range(self.max_tokens):
                    if generator.is_done():
                        break
                    generator.generate_next_token()
                return tokenizer.decode(generator.get_sequence(0)[len(tokens):])
            finally:
                del generator
        raise ValueError("Use a GGUF file or an ORT GenAI directory with configuration and tokenizer")

    def explain(self, prediction: dict | None, analysis: dict) -> dict:
        """Always return deterministic remediation even if local generation fails."""
        result = ThreatCopilot().explain(prediction, analysis)
        result["llm_guidance"] = None
        if self.model_path is None:
            result["fallback_reason"] = "No local language-model weights configured"
            return result
        facts = {"techniques": [m["id"] for m in result["mappings"]],
                 "evidence_type": "static indicators; execution unproven",
                 "model_status": prediction.get("model_status", "unknown") if prediction else "absent"}
        try:
            guidance = self._generate(facts)
            if not isinstance(guidance, str) or not guidance.strip():
                raise ValueError("Local model returned empty guidance")
            result["llm_guidance"] = guidance[:8000]
            result["reasoning_backend"] = "local GGUF" if self.model_path.suffix.lower() == ".gguf" else "local ONNX GenAI"
            result["text"] += "\n\nLocal model advice (unverified):\n" + result["llm_guidance"]
        except Exception as exc:
            result["fallback_reason"] = f"Local runner unavailable: {type(exc).__name__}"
        return result
