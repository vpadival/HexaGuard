"""Offline deterministic evidence mapper; does not impersonate a trained LLM."""
from __future__ import annotations


class ThreatCopilot:
    """Generate reviewable ATT&CK hypotheses and non-destructive guidance."""

    def explain(self, prediction: dict | None, analysis: dict) -> dict:
        """Map observed static indicators, never claim runtime behavior."""
        names = {item["name"] for item in analysis["apis"]}
        behaviors = set(analysis["behaviors"])
        mappings = []

        def add(identifier: str, name: str, tactic: str, evidence: str):
            mappings.append({"id": identifier, "name": name, "tactic": tactic,
                "evidence": evidence, "url": "https://attack.mitre.org/techniques/" +
                identifier.replace(".", "/") + "/"})

        if {"WriteProcessMemory", "CreateRemoteThread"} <= names:
            add("T1055", "Process Injection", "Defense Evasion / Privilege Escalation",
                "Memory-write and remote-thread API names co-occur; execution unproven")
        if {"NtUnmapViewOfSection", "SetThreadContext", "ResumeThread"} <= names:
            add("T1055.012", "Process Hollowing", "Defense Evasion / Privilege Escalation",
                "Unmapping and thread-context API names co-occur; process hollowing unproven")
        if "amsi_reference" in behaviors:
            add("T1562.001", "Impair Defenses: Disable or Modify Tools", "Defense Evasion",
                "AMSI-related text; a reference alone does not establish impairment")
        if behaviors & {"base64_wrapper", "string_concatenation", "variable_mutation"}:
            add("T1027", "Obfuscated Files or Information", "Defense Evasion",
                "Script syntax consistent with obfuscation, also common in benign scripts")
        if {"base64_wrapper", "iex_invocation"} <= behaviors:
            add("T1059.001", "PowerShell", "Execution",
                "Base64 and expression-invocation syntax co-occur; execution unproven")
        guidance = ["Review the original file and its source before allowing execution.",
                    "Correlate indicators with approved endpoint logs and known-good software."]
        if mappings or (prediction and prediction["is_malicious"]):
            guidance += ["Preserve the file hash and evidence for your security team.",
                         "If compromise is corroborated, follow your incident-response isolation procedure.",
                         "Verify endpoint protection health using your organization's approved tools."]
        else:
            guidance.append("No matched indicators does not establish that the file is safe.")
        score = "Model not loaded; probability unavailable" if prediction is None else (
            f"Model score: {prediction['threat_score']:.1%}; requires validation and analyst review")
        if prediction and prediction.get("is_demo"):
            score = "DEMO model: synthetic/untrained scores have no threat-detection meaning"
        return {"summary": score, "mappings": mappings, "remediation": guidance,
                "reasoning_backend": "deterministic offline evidence rules",
                "text": score + "\n\n" + "\n".join(guidance)}
