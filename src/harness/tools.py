"""Pi-style tool layer: read / write / edit / bash + cargo helpers.

Tools operate inside a per-rollout sandbox working directory (a checkout of
hyperswitch @ the task's parent commit). The same definitions are exposed as
OpenAI-style function schemas (for the model) and as executors (run by the runner).

Security (Risk R7/R11):
  * the runner is expected to launch this inside a gVisor/Firecracker sandbox with
    egress locked down and the hidden test files mounted READ-ONLY;
  * `protected_paths` are never writable here, and any mutation attempt is recorded
    so the reward layer can zero-reward tampering.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

# OpenAI-style tool schemas advertised to the model (Pi's default primitives + cargo).
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file (optionally a line range) from the working tree.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start": {"type": "integer"},
                    "end": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace an exact string in a file with new text (unique match).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old": {"type": "string"},
                    "new": {"type": "string"},
                },
                "required": ["path", "old", "new"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command in the working tree (sandboxed, no network).",
            "parameters": {
                "type": "object",
                "properties": {"cmd": {"type": "string"}},
                "required": ["cmd"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search the working tree for a regex; returns matching path:line.",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}, "glob": {"type": "string"}},
                "required": ["pattern"],
            },
        },
    },
]


@dataclass
class ToolBox:
    """Stateful executor bound to one rollout's working dir."""

    workdir: Path
    protected_paths: set[str] = field(default_factory=set)  # hidden tests, manifests
    tampered: bool = False
    closed_context: bool = False           # HS-Knowledge: forbid whole-repo grep
    allowed_files: set[str] = field(default_factory=set)

    def _resolve(self, path: str) -> Path:
        p = (self.workdir / path).resolve()
        if not str(p).startswith(str(self.workdir.resolve())):
            raise ValueError("path escapes workdir")
        return p

    def read_file(self, path: str, start: int | None = None, end: int | None = None) -> str:
        if self.closed_context and self.allowed_files and path not in self.allowed_files:
            return f"ERROR: closed-context task — {path} not in the provided file set."
        text = self._resolve(path).read_text(encoding="utf-8", errors="ignore")
        if start is None:
            return text
        lines = text.splitlines()
        return "\n".join(lines[(start or 1) - 1 : end or len(lines)])

    def edit_file(self, path: str, old: str, new: str) -> str:
        # normalize before the protected check — "./tests/x.rs" must not bypass "tests/x.rs"
        norm = str(Path(path)) if not path.startswith("/") else path
        if norm in self.protected_paths or path in self.protected_paths:
            self.tampered = True  # attempted to touch hidden tests/manifest -> reward=0
            return "ERROR: file is read-only (protected)."
        p = self._resolve(path)
        text = p.read_text(encoding="utf-8", errors="ignore")
        if text.count(old) != 1:
            return f"ERROR: 'old' must match exactly once (found {text.count(old)})."
        p.write_text(text.replace(old, new), encoding="utf-8")
        return "ok"

    def bash(self, cmd: str) -> str:
        # tamper guard: block edits to protected paths and VCS escapes via shell
        # ("chmod" included: `chmod +w tests/x.rs` would defeat the 0444 write-protection)
        lowered = cmd.replace(" ", "")
        if any(pp.replace(" ", "") in lowered for pp in self.protected_paths) and (
            ">" in cmd or "sed" in cmd or "tee" in cmd or "rm" in cmd or "chmod" in cmd
        ):
            self.tampered = True
            return "ERROR: command touches a protected path."
        if "git checkout" in cmd or "git restore" in cmd:
            self.tampered = True
            return "ERROR: VCS restore of gold files is not allowed."
        try:
            res = subprocess.run(
                cmd, shell=True, cwd=self.workdir, capture_output=True, text=True, timeout=600
            )
        except subprocess.TimeoutExpired:
            # a slow cargo build must NOT raise through the rollout loop and kill the
            # whole eval suite / training cycle — report it to the model as feedback
            return "ERROR: command timed out (600s). Use targeted builds (cargo check -p <crate>)."
        return (res.stdout + res.stderr)[:8000]

    def grep(self, pattern: str, glob: str = "") -> str:
        if self.closed_context:
            return "ERROR: closed-context task — whole-repo search disabled."
        args = ["grep", "-rnE", pattern, "."]
        if glob:
            args = ["grep", "-rnE", "--include", glob, pattern, "."]
        res = subprocess.run(args, cwd=self.workdir, capture_output=True, text=True)
        return res.stdout[:8000]

    def dispatch(self, name: str, args: dict) -> str:
        fn = getattr(self, name, None)
        if fn is None:
            return f"ERROR: unknown tool {name}"
        return fn(**args)


def nextest_json(workdir: Path, package: str | None) -> dict:
    """Run cargo nextest with structured JSON output (NEVER log-scrape — Risk R11)."""
    import os
    cmd = ["cargo", "nextest", "run", "--message-format", "libtest-json"]
    if package:
        cmd += ["-p", package]
    # libtest-json is gated behind this env var — without it nextest ERRORS, zero events
    # parse, and the empty passed/failed lists silently zero the test reward everywhere.
    env = {**os.environ, "NEXTEST_EXPERIMENTAL_LIBTEST_JSON": "1",
           "PATH": os.path.expanduser("~/.cargo/bin") + os.pathsep + os.environ.get("PATH", "")}
    try:
        res = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True,
                             timeout=1800, env=env)
    except subprocess.TimeoutExpired:
        return {"exit_code": -1, "passed": [], "failed": [], "error": "nextest timed out"}
    passed, failed = [], []
    for line in res.stdout.splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "test" and ev.get("event") in ("ok", "failed"):
            (passed if ev["event"] == "ok" else failed).append(ev.get("name", "?"))
    out = {"exit_code": res.returncode, "passed": passed, "failed": failed}
    if res.returncode != 0 and not passed and not failed:
        # nonzero exit with ZERO parsed events = the run itself broke (toolchain/flag/build
        # error), not "all tests failed" — surface it so callers don't read it as empty-pass.
        out["error"] = (res.stderr or res.stdout)[-500:]
    return out
