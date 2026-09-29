"""Optional real transformer export test, using tiny local random weights only."""
from pathlib import Path

import pytest


def test_local_transformer_export_and_tokenizer(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import BertTokenizerFast, DistilBertConfig, DistilBertForSequenceClassification
    from scripts.export_threat_model import export_models
    from src.engine.qnn_detector import HexaGuardDetector

    source = tmp_path / "tiny-source"
    source.mkdir()
    vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world", "test"]
    (source / "vocab.txt").write_text("\n".join(vocabulary), encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(source / "vocab.txt"))
    tokenizer.save_pretrained(source)
    config = DistilBertConfig(vocab_size=len(vocabulary), max_position_embeddings=32,
                             n_layers=1, n_heads=2, dim=16, hidden_dim=32, num_labels=2)
    DistilBertForSequenceClassification(config).save_pretrained(source)
    output = tmp_path / "exported"
    card = export_models(output, source=str(source), length=16)
    assert card["tokenizer"] == "hf-local-v1", card
    detector = HexaGuardDetector(str(output / "threat_detector_int8.onnx"), backend="cpu")
    result = detector.predict("hello world " * 30)
    assert result["windows"] > 1
    assert result["is_demo"] and result["is_malicious"] is None
    assert 0 <= result["threat_score"] <= 1
