from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import tempfile

from .config import resolve_command


@dataclass(frozen=True)
class ProviderAvailability:
    name: str
    command: str | None
    available: bool
    reason: str


@dataclass(frozen=True)
class ProviderResult:
    provider: str
    command: list[str]
    returncode: int
    stdout: str
    stderr: str


def detect_providers(config: dict) -> dict[str, ProviderAvailability]:
    claude_cfg = config["providers"]["claude"]
    codex_cfg = config["providers"]["codex"]

    claude_command = resolve_command(claude_cfg["command"], ["claude"])
    codex_command = resolve_command(codex_cfg["command"], ["codex.cmd", "codex.exe", "codex"])

    return {
        "claude": ProviderAvailability(
            name="claude",
            command=claude_command,
            available=bool(claude_command) and bool(claude_cfg["enabled"]),
            reason="ok" if claude_command else "command not found",
        ),
        "codex": ProviderAvailability(
            name="codex",
            command=codex_command,
            available=bool(codex_command) and bool(codex_cfg["enabled"]),
            reason="ok" if codex_command else "command not found",
        ),
    }


def run_provider_prompt(
    provider: str,
    command: str,
    prompt: str,
    cwd: str,
    model: str = "",
    allow_edits: bool = False,
) -> ProviderResult:
    output_file: str | None = None
    if provider == "claude":
        args = [command, "-p"]
        args.extend(["--permission-mode", "acceptEdits" if allow_edits else "plan"])
        if model:
            args.extend(["--model", model])
        args.append(prompt)
    elif provider == "codex":
        sandbox = "workspace-write" if allow_edits else "read-only"
        temp_handle = tempfile.NamedTemporaryFile(prefix="orchestrator-codex-", suffix=".txt", delete=False)
        temp_handle.close()
        output_file = temp_handle.name
        args = [command, "exec", "-C", cwd, "-s", sandbox, "--color", "never", "-o", output_file]
        if model:
            args.extend(["-m", model])
        args.append(prompt)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    result = subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    stdout = result.stdout.strip()
    if output_file:
        output_path = Path(output_file)
        if output_path.exists():
            file_text = output_path.read_text(encoding="utf-8").strip()
            if file_text:
                stdout = file_text
            output_path.unlink(missing_ok=True)

    return ProviderResult(
        provider=provider,
        command=args,
        returncode=result.returncode,
        stdout=stdout,
        stderr=result.stderr.strip(),
    )
