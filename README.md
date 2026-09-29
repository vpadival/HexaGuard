# HexaGuard

Offline endpoint triage and threat disassembly for Snapdragon HP PCs.

HexaGuard inspects user-submitted scripts and binaries, runs local ONNX inference,
and presents static evidence with MITRE ATT&CK hypotheses and remediation guidance.
The desktop runtime does not execute submitted files or call cloud services.

**A working INT8 demo model is now bundled.** It has deterministic synthetic
weights and is **not trained to detect malware**. Its scores exercise the complete
pipeline; they are explicitly labeled DEMO, and `is_malicious` is `None`. No
detection accuracy or verified Snapdragon performance is claimed. A fine-tuned,
independently validated classifier is still needed for operational decisions.

## Quick start

From the repository root in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m src.app.main_gui
```

The GUI loads `assets/models/threat_detector_int8.onnx` by default. Paste text,
open a file, or drop a local file. Select `auto` for scripts/PEs; choose `x64` or
`arm64` explicitly for raw instruction bytes. `--static-only` disables inference.
The display shows measured local inference latency and process RAM. DEMO scores
are never presented as threat probabilities.

## 1. Technical Implementation

### Bundled classifier and one-command export

The repository includes:

| Artifact | Purpose |
| --- | --- |
| `assets/models/threat_detector.onnx` | Floating-point synthetic reference graph |
| `assets/models/threat_detector_int8.onnx` | Dynamically quantized INT8 CPU demo |
| `assets/models/model_card.json` | Provenance, demo status, tokenizer, labels and SHA-256 hashes |

Rebuild without network access:

```powershell
python scripts/export_threat_model.py --synthetic
```

The fallback uses seeded random weights, MatMul layers and a class-1 sigmoid.
It is input-dependent, with class convention 0=Benign and 1=Malicious; those names
do not imply learned semantics. Both graphs are checked with ONNX checker and
executed on CPU before publication. Export verifies actual INT8 weight tensors.
Successful generation requires installed dependencies and writable disk space.

To export a lightweight Hugging Face sequence classifier:

```powershell
python -m pip install -r requirements-models.txt
python scripts/export_threat_model.py
python scripts/export_threat_model.py --source distilbert-base-uncased --allow-download
python scripts/export_threat_model.py --source C:\Models\MyFineTunedThreatClassifier
```

Without `--allow-download`, loading is local/cache-only. Missing dependencies,
offline cache misses, authentication failures, or export errors fall back to the
synthetic graph and record the exception type in the model card. Existing output
models are replaced only after staging and validation succeed. `--output` selects
a separate directory. A tokenizer directory is also saved for transformer exports.
The bundled synthetic export does not require transformers or torch.

`distilbert-base-uncased` has no trained threat-classification head. Adding a
binary head does not train it. Base models and checkpoints without an explicit
Benign/Malicious label map are marked demo. A checkpoint declaring
`id2label={0: "Benign", 1: "Malicious"}` is marked external/unvalidated, never
automatically certified. Other label orders require retraining/re-exporting with
the correct class convention. Respect upstream weight licenses and access terms.

### Tokenizers and model contracts

Both supported exports return one float32 probability `[1,1]`. Input dimensions
are fixed `[1,N]`. The ONNX metadata identifies preprocessing:

* `byte-v1`: one integer input, raw byte plus one, zero padding, vocabulary 257.
  Pasted text uses UTF-8; file content remains raw bytes.
* `hf-local-v1`: `input_ids` and `attention_mask`, with the export's adjacent
  `tokenizer/` folder. It loads locally with remote code disabled. Text is decoded
  from UTF-8 or UTF-16 and tokenized with overflow windows; binary text decoding
  is lossy, so this contract is primarily appropriate for script/text classifiers.

The detector takes the maximum window score; this is not calibrated file-level
probability. Inputs are limited to 8 MiB and 4096 model windows. A non-demo model's
default threshold is 0.7, subject to independent calibration. Preserve model
metadata and tokenizer assets in every deployment. QNN exports can alter tensor
types and strip metadata: verify the contract before copying metadata from the
source. Metadata alone does not make an incompatible graph compatible.

### QNN HTP backend and zero CPU graph offloading

The detector attempts `QNNExecutionProvider`, `backend_path=QnnHtp.dll` and burst
mode. QNN sessions disable CPU graph fallback and runtime fallback. Unsupported
graphs fail strict QNN mode; automatic mode warns and creates a separate CPU
session. The NPU badge requires successful strict QNN inference. Parsing,
tokenization, guidance and UI work still run on the CPU.

The dynamically quantized INT8 artifact is intended for CPU execution. HTP
compilation uses static calibration of the floating-point source. Context binaries
are device-specific; a `.bin` cannot be loaded as a portable ONNX graph.

```powershell
python -m src.app.main_gui --backend cpu
python -m src.app.main_gui --model assets\models\qnn_bundle\model.onnx --cpu-model assets\models\threat_detector_int8.onnx --backend auto --qnn-backend C:\Qualcomm\QNN\QnnHtp.dll
```

Replace the example paths with installed artifacts. Use `--backend qnn` to fail
instead of falling back. Preserve the ONNX context wrapper's referenced binary.

### Qualcomm AI Hub cloud compilation and profiling

The SDK integration uses `qai_hub.Client` with an in-memory `ClientConfig` API
token. Set `QAI_HUB_API_TOKEN` securely in your environment, or use `--api-token`
(the environment avoids exposing a token through command history). The script
does not persist or print tokens. Live mode requires `--allow-upload` because
models and calibration data leave the machine. Runtime inference stays offline.

Local demo, requiring no account or network:

```powershell
python scripts/compile_ai_hub.py --dry-run
```

Without a token, the script also defaults to mock mode. `--live` forces live mode
and fails if credentials are unavailable. A live failure never switches to mock
success. Mock mode writes `profile_mock.json`, sets `simulated: true`, creates no
cloud job, and **does not create a QNN binary**.

For real compilation, prepare a representative NPZ calibration set using the
source model's tokenizer. Each input array has shape `[samples,1,N]` and exactly
the source input dtype. Use the `tokens` key for byte-v1; use both `input_ids` and
`attention_mask` for transformer exports. Every input must have the same sample
count. Use authorized, non-sensitive calibration data.

```powershell
python scripts/compile_ai_hub.py --live --allow-upload --calibration C:\Models\calibration.npz
```

The live workflow:

1. Confirms `Snapdragon X Elite CRD` is accessible to the account.
2. Uploads the dynamic INT8 CPU artifact and the floating-point source.
3. Requests static INT8 weights/activations using the SDK quantization job.
4. Submits `--target_runtime qnn_dlc --compute_unit npu`, then links the DLC
   into `assets/models/threat_detector_qnn.bin`.
5. Downloads an ONNX context wrapper and profiles the context on NPU and the
   dynamic INT8 baseline on CPU.
6. Saves job IDs, source hash, raw profiles and `profile_hardware.json`.

The requested legacy string `--target_runtime qnn_lib_aarch64_windows --precision
int8` is not a supported option in the current official API. A QNN library is also
not a context binary. This implementation uses the documented DLC compile/link
flow instead. Quantization is applied explicitly rather than through an invented
precision flag. Cloud operator/device/account availability can still cause failure.

Median latency is computed from SDK `all_inference_times` in microseconds. If the
SDK only supplies an estimate, it is stored separately, not relabeled a median.
Peak memory and ranges are retained in bytes, with missing fields set to null.
`htp_only_verified` is true only if reported execution units contain solely NPU;
missing placement information is not considered verification. Check raw profiles
before asserting zero CPU offloading. The CPU and HTP graphs use different
quantization schemes; validate numerical parity before comparing performance.

### Hardware performance metrics

| Source | NPU median | CPU median | NPU peak memory | CPU peak memory |
| --- | --- | --- | --- | --- |
| **Simulated demo fixture** | 4.2 ms | 38.6 ms | 64 MiB | 128 MiB |
| Verified Snapdragon run | Not yet measured | Not yet measured | Not yet measured | Not yet measured |

The demo values are illustrative constants, not predictions, device measurements,
or evidence of a speedup. Real results are written only after successful hardware
jobs. No temperature/power claim is derived from latency.

To measure locally on an equipped Snapdragon PC:

```powershell
python scripts/run_benchmark.py --cpu-model assets\models\threat_detector_int8.onnx --qnn-model assets\models\qnn_bundle\model.onnx --input tests\sample_benign.ps1
```

It performs five warmups and 100 scans in separate backend processes, reporting
mean/P99 end-to-end latency, throughput and sampled host RSS delta. RSS excludes
NPU allocations; startup is excluded from latency. NPU errors remain unavailable.
Thermal footprint remains unmeasured without external instrumentation.

### Tier-2 offline guidance

`OfflineGuidance` in `src/engine/offline_guidance.py` connects the existing
`ThreatCopilot` evidence mapper to optional local language models. No weights are
downloaded implicitly. The rules always provide guidance, including AMSI-related
references (T1562.001), injection (T1055), process-hollowing API combinations
(T1055.012), and Base64/IEX syntax (T1027/T1059.001). These are hypotheses based on
static text, not proof of behavior. Benign scripts and comments can match.

For GGUF, install `llama-cpp-python` for your native architecture and select a
local instruction-tuned GGUF, such as a compatible quantization of Llama-3.2-1B
Instruct or Qwen2.5-Coder-1.5B Instruct:

```powershell
python -m pip install llama-cpp-python
python -m src.app.main_gui --guidance-model C:\Models\assistant.gguf
```

For ONNX, install `onnxruntime-genai` and supply a compatible **GenAI model
directory**, including `genai_config.json`, tokenizer files and quantized weights.
An arbitrary standalone ONNX file is insufficient. Add `chat_template.txt` with
the model's exact instruction/chat formatting and a literal `{prompt}` insertion
point. This avoids silently applying one model family's special tokens to another.

```powershell
python -m pip install onnxruntime-genai
python -m src.app.main_gui --guidance-model C:\Models\qwen-genai
```

Install only the runner you use; `requirements-guidance.txt` lists both optional
adapters. Native wheel availability varies. Generation receives bounded technique
IDs and status, not raw submitted content; it has no tool execution privileges.
Generated advice is shown as unverified alongside the original rule evidence.
Missing weights, configuration errors and generation exceptions retain the rule
fallback. Local LLM weights are not bundled because of size and licensing.

## 2. Application Use Case & Innovation

The intended use is private, automatic triage of a user-selected file on a
Snapdragon laptop. Capstone supports x64/ARM64; PE headers select architecture
and imports provide API indicators. Raw disassembly requires an explicit ISA.
Inspection is capped at 512 instructions and 64 KiB per executable region.
Highlighted call instructions are not resolved suspicious API invocations.

The system provides transparent evidence, an offline model interface and optional
local guidance. It does not continuously monitor, block, quarantine or modify
processes. “Autonomous” describes submitted-file analysis, not incident response.

## 3. Deployment & Accessibility on HP Snapdragon PCs

Use native ARM64 Python 3.10+ on Windows 11 and current OEM drivers. Verify native
PyQt6, inference and optional LLM wheels for the chosen Python version. The default
`onnxruntime` package is for CPU development; replace it with a vendor-supported
QNN-enabled distribution for native Windows ARM64. Do not install competing
distributions supplying the same `onnxruntime` module in one environment.
Use matched QNN HTP libraries, dependencies and drivers from trusted packages.

```powershell
python -c "import platform, onnxruntime as ort; print(platform.machine()); print(ort.get_available_providers())"
```

For air-gapped deployment, prepare a vetted wheelhouse on a matching connected
machine, transfer packages plus model/tokenizer files, and install with
`pip install --no-index --find-links wheelhouse -r requirements.txt`. Freeze the
tested versions and record all model hashes. No AI Hub credentials are needed on
the runtime PC. Optional Hugging Face export and AI Hub builds are connected
preparation steps, separate from offline analysis.

## Tests and release status

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests cover bundled INT8 CPU fallback, model hashes, offline export fallback,
profile units, mock CLI behavior, failed jobs, fake-SDK live call sequencing,
guidance fallbacks, disassembly, and GUI worker flow. Fake SDK tests validate
integration logic; they are not cloud/device verification. The malicious-named
PowerShell fixture is inert comment text and does not implement an AMSI bypass.

This is a working reference application, not a production-validated protection
product. Operational deployment requires training/accuracy and false-positive
evaluation, threshold calibration, quantization parity testing, actual X Elite/
X Plus benchmarks, signed artifacts, pinned dependencies and parser fuzz testing.
Native parsers run in a worker thread rather than an OS sandbox; isolate hostile
input processing in a restricted process before production service use.

## Official references

* [AI Hub API and supported options](https://workbench.aihub.qualcomm.com/docs/hub/api.html)
* [AI Hub compilation and context linking](https://workbench.aihub.qualcomm.com/docs/hub/compile_examples.html)
* [AI Hub quantization](https://workbench.aihub.qualcomm.com/docs/hub/quantize_examples.html)
* [ONNX Runtime QNN provider](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html)
* [ONNX Runtime quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
* [ONNX Runtime GenAI Python API](https://onnxruntime.ai/docs/genai/api/python.html)
* [MITRE Process Injection](https://attack.mitre.org/techniques/T1055/)
* [MITRE Impair Defenses](https://attack.mitre.org/techniques/T1562/001/)
