# HexaGuard

Offline Endpoint & Threat Disassembly Engine for Snapdragon HP PCs.

HexaGuard is a read-only desktop triage application for Windows 11 on ARM64,
including Snapdragon X Elite / X Plus systems. It combines bounded static
inspection, a supplied ONNX classifier, and local evidence-based guidance.
Files are never executed and the runtime makes no cloud requests.

**Release status:** working reference implementation, not a validated production
endpoint protection product. No trained threat classifier or quantized language
model is bundled. Static analysis works immediately; probabilities require your
own compatible trained model. No accuracy, sub-10 ms latency, NPU throughput,
or thermal claim is made without measurement. It does not continuously monitor,
block, quarantine, or remediate processes. “Autonomous” here means local automatic
analysis of a user-submitted file, not an autonomous response agent.

## 1. Technical Implementation

### QNN HTP backend and CPU fallback

`HexaGuardDetector` requests `QNNExecutionProvider` with `QnnHtp.dll` and
`htp_performance_mode=burst`. QNN sessions set
`session.disable_cpu_ep_fallback=1`, so unsupported graph nodes fail initialization.
Runtime fallback is also disabled. With `--backend auto`, initialization failure
logs a warning and opens a separate CPU session. `--backend qnn` fails instead.
The NPU badge appears only after a successful strict QNN prediction.

“Zero CPU offloading” applies to inference graph placement only. File reading,
tokenization, static analysis, GUI rendering, and guidance run on the CPU. Some
exported models contain CPU-dependent I/O wrappers and will intentionally fail
strict mode. Use a compatible graph and verify node placement with vendor
profiling tools before making performance claims.

A context-only ONNX model cannot run on CPU. Supply its portable ONNX equivalent
with `--cpu-model`; preserve the adjacent context binary and its relative path.
The bare `.bin` artifact is not an ONNX Runtime model.

### Model and tokenizer contract

The application intentionally refuses arbitrary models whose preprocessing is
unknown. Supply a trusted model with:

* One input, `int32` or `int64`, fixed shape `[1,N]`, `1 <= N <= 1048576`.
* Byte tokenizer `byte-v1`: UTF-8 for pasted text; raw bytes for files; byte value
  plus one is the token ID; zero is padding. Vocabulary size is at least 257.
* ONNX metadata `hexaguard.tokenizer=byte-v1` on each deployment artifact.
* One float32 output `[1,1]`, a finite malicious-class probability in `[0,1]`.
* Training, calibration, and evaluation using exactly this preprocessing.

For long inputs the detector runs nonoverlapping windows and takes the maximum
score. This aggregate is a model score, not a calibrated file-level probability.
The default decision threshold is 0.7. Maximum input is 8 MiB and maximum windows
is 4096; larger jobs fail explicitly. Patterns spanning window boundaries can be
missed by the model. Static rules examine the bounded complete input.

Model exports may lose custom metadata. Restore it locally only after checking
that the tokenizer, vocabulary, and tensor semantics still match. Changing the
metadata alone does not make an incompatible model compatible.

### Disassembly and local reasoning

Capstone decodes x64 and ARM64. PE machine headers select the ISA, executable
sections supply bytes, and imports supply API indicators. Raw code requires an
explicit ISA. Inspection is capped at 512 instructions and 64 KiB per region;
linear decoding cannot recover all control flow or dynamically resolved calls.
Call instructions are highlighted separately from suspicious API names; there
is no claim that a highlighted call invokes a particular API.

PowerShell rules flag Base64 wrappers, concatenation, variable mutation, and AMSI
references, including UTF-16 text. They do not decode or execute payloads.
Comments and benign administration scripts can match these rules.
`ThreatCopilot` is a deterministic offline evidence mapper, not a quantized LLM.
It returns ATT&CK hypotheses, evidence, and non-destructive remediation guidance.
T1055 is a technique, and T1562.001 is a sub-technique; their tactics are reported
separately. The bundled mapping is a static snapshot, not a live ATT&CK feed.

### Quantization and AI Hub

`compile_ai_hub.py` provides an **online build-time** workflow. AI Hub uploads
models and calibration inputs; it is separate from the air-gapped runtime.
Never use sensitive endpoint samples as calibration data without authorization.

`w8a8` uses the SDK quantization job with INT8 weights and activations. `w4a16`
accepts an externally quantized compatible artifact via `--prequantized`;
the script does not invent unsupported generic 4-bit quantization flags.
Compatibility depends on graph operators, SDK/runtime versions and target device.
The script does not verify a pre-quantized artifact's actual bit widths: review
encodings and compiler output. `qai-hub-models` is an optional model-development
tool, not a source of a trained HexaGuard threat classifier.

## 2. Application Use Case & Innovation

The intended use is private, local triage on a Snapdragon laptop: drop a script
or binary, inspect readable evidence and hex bytes, then correlate findings with
your incident-response workflow. No endpoint content needs to leave the PC.
Local model scoring can reduce dependence on online services once an appropriate
classifier has been trained and validated. Human review remains necessary.

## 3. Deployment & Accessibility on HP Snapdragon PCs

Use native ARM64 Python 3.10+ on Windows 11 with current OEM drivers. Confirm
native wheels exist for your chosen Python version before deployment; package
availability differs across Python and processor architectures. In particular,
PyQt6 and QNN-enabled ONNX Runtime require compatible native builds.

From the repository root in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m src.app.main_gui
```

This starts static mode without a model. Paste text, use Open file, or drag one
local file onto the window. Use `auto` for PE/script inputs; select `x64` or
`arm64` only for raw instruction bytes. The interface displays measured inference
latency when a model is loaded, host CPU utilization, and process RAM.

The default `onnxruntime` dependency supports CPU development. For NPU deployment,
replace it with a Qualcomm-compatible QNN-enabled ONNX Runtime distribution for
your exact native Python/Windows architecture, following the official QNN guide.
Do not install two distributions that both provide the `onnxruntime` module into
the same environment. Deploy the matching QNN HTP backend, dependent libraries,
and device drivers from trusted vendor packages. A DLL name alone is insufficient.

```powershell
python -c "import platform, onnxruntime as ort; print(platform.machine()); print(ort.get_available_providers())"
python -m src.app.main_gui --model models\threat.onnx --backend cpu
python -m src.app.main_gui --model models\qnn\threat.onnx --cpu-model models\threat.onnx --backend auto --qnn-backend C:\Qualcomm\QNN\lib\arm64x-windows-msvc\QnnHtp.dll
```

The example SDK path must be replaced with the installed, trusted backend path.
Use `--backend qnn` to enforce HTP-only inference. A clean CPU fallback requires
a portable equivalent model. No model is downloaded automatically.

For air-gapped deployment, prepare a wheelhouse on a connected machine matching
the target architecture and Python version, transfer vetted packages and model
artifacts, then install with `pip install --no-index --find-links wheelhouse -r
requirements.txt`. Freeze the tested environment and record model hashes before
release. AI Hub is not needed at runtime.

### Optional model compilation

Configure AI Hub credentials using its official SDK setup; do not store tokens
in this repository. Confirm the device name is available to your account.
Calibration NPZ contains one array per model input with shape `[samples,1,N]`
and the model's integer dtype; use representative, non-sensitive samples.

```powershell
python scripts\compile_ai_hub.py models\threat.onnx --calibration models\calibration.npz --quantization w8a8 --allow-upload
python scripts\compile_ai_hub.py models\threat.aimet --quantization w4a16 --prequantized --allow-upload
```

Outputs include a context binary, an ONNX deployment artifact (which may be a
directory/archive), and build metadata. Preserve the downloaded bundle structure.
Revalidate the tensor/tokenizer contract and compare accuracy with the portable
model before use. Unsupported graphs or account/device access failures are errors.

### Benchmarking

```powershell
python scripts\run_benchmark.py --cpu-model models\threat.onnx --qnn-model models\qnn\threat.onnx --input tests\sample_benign.ps1
```

Each backend runs in a fresh process, with five warmups and 100 measured scans.
The table reports end-to-end mean/P99 latency, files per second, and sampled RSS
delta from before model initialization. RSS excludes NPU allocations and is not
a precise peak-memory measurement. Compilation/session startup is excluded from
timing. Thermal footprint is **not measured**; no defensible temperature or power
estimate can be derived from latency alone. Use physical power/thermal telemetry
for those measurements. NPU errors are reported as unavailable, never CPU results.

### Tests and release validation

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests use an explicitly synthetic constant-output ONNX graph solely to verify
the inference contract and fallback behavior. It is not shipped as a classifier.
`sample_malicious.ps1` is an inert comment-based indicator fixture with a harmless
Write-Output statement; it does not implement an AMSI bypass.

Before production use: train and validate a classifier on representative held-out
data, calibrate thresholds and file aggregation, evaluate adversarial evasion and
false positives, test X Elite/X Plus hardware and operator placement, sign and
hash deployment artifacts, pin a tested dependency set, and perform malformed
input/fuzz testing. Native parsers currently run in a worker thread, not an OS
sandbox; isolate analysis in a restricted process before processing hostile files
in a production service. This repository does not replace endpoint protection.

## Official implementation references

* [ONNX Runtime QNN Execution Provider](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html)
* [Qualcomm AI Hub compilation](https://workbench.aihub.qualcomm.com/docs/hub/compile_examples.html)
* [Qualcomm AI Hub quantization](https://workbench.aihub.qualcomm.com/docs/hub/quantize_examples.html)
* [MITRE T1055 Process Injection](https://attack.mitre.org/techniques/T1055/)
* [MITRE T1562.001 Impair Defenses](https://attack.mitre.org/techniques/T1562/001/)
* [MITRE T1027 Obfuscated Files or Information](https://attack.mitre.org/techniques/T1027/)
