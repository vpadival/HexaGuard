"""Bounded, non-executing PE, raw instruction and script inspection.

API names are indicators, not proof of executed calls. This is a disassembler,
not a source-code decompiler or a behavioral sandbox.
"""
from __future__ import annotations

import re

MAX_BYTES = 8 * 1024 * 1024
APIS = ("VirtualAlloc", "VirtualAllocEx", "WriteProcessMemory",
        "CreateRemoteThread", "AmsiScanBuffer", "AmsiInitialize",
        "NtUnmapViewOfSection", "SetThreadContext", "ResumeThread")


def decode_text(data: bytes) -> str:
    """Decode common PowerShell encodings without evaluating content."""
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", errors="replace")
    if len(data) > 8 and data[1:200:2].count(0) > len(data[1:200:2]) // 2:
        return data.decode("utf-16-le", errors="replace")
    return data.decode("utf-8-sig", errors="replace")


class DisassemblyAnalyzer:
    """Inspect up to 8 MiB and 512 instructions; never run an input file."""

    def analyze(self, data: bytes | str, architecture: str = "auto") -> dict:
        """Return evidence and warnings. Raw code requires an explicit ISA."""
        if architecture not in ("auto", "x64", "arm64"):
            raise ValueError("Architecture must be auto, x64, or arm64")
        raw = data.encode("utf-8") if isinstance(data, str) else data
        if not isinstance(raw, bytes) or len(raw) > MAX_BYTES:
            raise ValueError("Input must be bytes or text, at most 8 MiB")
        text = decode_text(raw)
        result = {"format": "text/data", "architecture": None, "apis": [],
                  "behaviors": [], "instructions": [], "warnings": []}
        for name in APIS:
            if name.lower() in text.lower() or name.encode("utf-16-le") in raw:
                result["apis"].append({"name": name, "source": "string"})
        patterns = {
            "base64_wrapper": r"(?i)(?:-enc(?:odedcommand)?\b|frombase64string\s*\()",
            "string_concatenation": r"['\"]\s*\+\s*['\"]",
            "variable_mutation": r"(?i)\$[\w]+\s*\+=|set-variable\b",
            "amsi_reference": r"(?i)amsi(?:scanbuffer|initialize|utils|initfailed)",
            "iex_invocation": r"(?i)\b(?:iex|invoke-expression)\b",
        }
        result["behaviors"] = [key for key, regex in patterns.items() if re.search(regex, text)]
        regions = []
        if raw.startswith(b"MZ"):
            import pefile
            pe = None
            try:
                pe = pefile.PE(data=raw, fast_load=True)
                result["format"] = "PE"
                architecture = {0x8664: "x64", 0xAA64: "arm64"}.get(pe.FILE_HEADER.Machine)
                result["architecture"] = architecture
                pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]])
                for entry in getattr(pe, "DIRECTORY_ENTRY_IMPORT", []):
                    for item in entry.imports:
                        name = (item.name or b"").decode("ascii", errors="replace")
                        if name in APIS:
                            result["apis"].append({"name": name, "source": "PE import",
                                                   "address": hex(item.address)})
                for section in pe.sections:
                    if section.Characteristics & 0x20000000:
                        regions.append((section.get_data()[:65536],
                                        pe.OPTIONAL_HEADER.ImageBase + section.VirtualAddress))
                if architecture is None:
                    result["warnings"].append("Unsupported PE machine; instructions omitted")
            except pefile.PEFormatError as exc:
                result["warnings"].append(f"Malformed PE: {exc}")
                architecture = None
            finally:
                if pe is not None:
                    pe.close()
        elif architecture != "auto":
            result.update(format="raw code", architecture=architecture)
            regions = [(raw[:65536], 0)]
        if architecture in ("x64", "arm64") and regions:
            import capstone
            arch, mode = ((capstone.CS_ARCH_X86, capstone.CS_MODE_64) if architecture == "x64"
                          else (capstone.CS_ARCH_ARM64, capstone.CS_MODE_ARM))
            decoder = capstone.Cs(arch, mode)
            for code, base in regions:
                remaining = 512 - len(result["instructions"])
                if remaining <= 0:
                    break
                for ins in decoder.disasm(code, base, count=remaining):
                    result["instructions"].append({"address": hex(ins.address),
                        "mnemonic": ins.mnemonic, "operands": ins.op_str,
                        "is_call": ins.mnemonic in ("call", "bl", "blr")})
            result["warnings"].append("Bounded linear disassembly; call targets are not resolved")
        return result
