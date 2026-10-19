# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run one LAB task through Claude Code, the claude twin of run_codex_lab.py.

Writes config.json, claude_session.jsonl, output/ and metrics.json last as the completion
record, so a dead model chain is never cached as a finished rollout. The adapter has already
built the process environment (isolated CLAUDE_CONFIG_DIR, auth routing, LAB PATH).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from harness2.substrate.claude import CLAUDE_ROOT_ENTRIES, ClaudeSubstrate, parse_events


def run_dir_for(results_root: Path, run_id: str) -> Path:
    """<results_root>/<run_id>, with path escapes refused."""
    rid = str(run_id).strip()
    if not rid or rid.startswith("/") or ".." in rid.split("/"):
        raise SystemExit(f"bad --run-id {run_id!r}: must be a relative, escape-free id")
    root = Path(results_root).resolve()
    d = (root / rid).resolve()
    if root not in d.parents:
        raise SystemExit(f"run id escapes the results root: {run_id!r}")
    return d


def task_json(lab_root: Path, task_id: str) -> dict:
    """task.json WITHOUT `criteria`: the agent never sees the grading rubric."""
    data = json.loads((lab_root / "tasks" / task_id / "task.json").read_text(encoding="utf-8"))
    data.pop("criteria", None)
    return data


def build_prompt(lab_root: Path, task_id: str, workspace: Path) -> str:
    """The assignment message, worded exactly as the codex runner words it."""
    instructions = str(task_json(lab_root, task_id).get("instructions", "")).strip()
    return (
        f"=== WORKSPACE ===\n{workspace}\n\n"
        f"=== DOCUMENTS (read-only) ===\n{workspace / 'documents'}\n\n"
        f"=== OUTPUT DIRECTORY (deliverables go here) ===\n{workspace / 'output'}\n\n"
        f"=== ASSIGNMENT ===\n{instructions}"
    )


def install_harness(ws: Path, harness_dir: Path | None, read_doc: Path) -> list[str]:
    """Merge the materialised harness into the workspace root; return the installed paths.

    The runtime's read_doc.py is installed last so no harness file can shadow it.
    """
    installed: list[str] = []
    if harness_dir is not None:
        for entry in CLAUDE_ROOT_ENTRIES:
            src = harness_dir / entry
            if src.is_file():
                shutil.copy2(src, ws / entry)
                installed.append(entry)
            elif src.is_dir():
                for f in sorted(src.rglob("*")):
                    if not f.is_file() or "__pycache__" in f.parts:
                        continue
                    rel = f.relative_to(harness_dir)
                    (ws / rel).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, ws / rel)
                    installed.append(str(rel))
    (ws / "scripts").mkdir(parents=True, exist_ok=True)
    if not read_doc.is_file():
        raise SystemExit(
            f"runtime document reader missing: {read_doc} -- without it every binary "
            f"document 'read' silently returns noise"
        )
    shutil.copy2(read_doc, ws / "scripts" / "read_doc.py")
    return installed


def parse_session(session_path: Path) -> dict:
    """Summarise a stream-json session for metrics.json."""
    events = [e for e in parse_events(session_path) if e.get("type") != "_read_error"]
    tool_uses: list[dict] = []
    result: dict = {}
    api_errors = 0
    for ev in events:
        if ev.get("type") == "assistant" and isinstance(ev.get("message"), dict):
            content = ev["message"].get("content")
            if isinstance(content, list):
                tool_uses += [
                    b for b in content if isinstance(b, dict) and b.get("type") == "tool_use"
                ]
        elif ev.get("type") == "system" and ev.get("subtype") == "api_error":
            api_errors += 1
        elif ev.get("type") == "result":
            result = ev
    usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
    details = usage.get("output_tokens_details")
    details = details if isinstance(details, dict) else {}

    def n(v) -> int:
        try:
            return int(v or 0)
        except (TypeError, ValueError):
            return 0

    cache_read = n(usage.get("cache_read_input_tokens"))
    return {
        "n_events": len(events),
        "tool_calls": len(tool_uses),
        "commands": [
            str(b.get("input", {}).get("command"))
            for b in tool_uses
            if b.get("name") == "Bash" and isinstance(b.get("input"), dict)
        ],
        "errors": api_errors + (1 if result.get("is_error") else 0),
        "turns": n(result.get("num_turns")),
        "final_text": str(result.get("result") or ""),
        "input_tokens": n(usage.get("input_tokens"))
        + n(usage.get("cache_creation_input_tokens"))
        + cache_read,
        "cached_input_tokens": cache_read,
        "output_tokens": n(usage.get("output_tokens")),
        "reasoning_output_tokens": n(details.get("thinking_tokens")),
        "cost_usd": result.get("total_cost_usd"),
    }


def main(args) -> int:
    lab_root = Path(args.lab_root).resolve()
    if not os.environ.get("CLAUDE_CONFIG_DIR"):
        raise SystemExit("CLAUDE_CONFIG_DIR unset; the rollout would load the operator's setup")
    label = args.run_id.replace("/", "__")
    run_dir = run_dir_for(Path(args.results_root), args.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(args.scratch).resolve() / "lab-cc" / f"{label}__{os.getpid()}"
    ws = tmp_root / "ws"

    def _die(signum, _frame):
        raise SystemExit(128 + signum)

    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(s, _die)

    docs_src = lab_root / "tasks" / args.task / "documents"
    started = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    try:
        shutil.rmtree(tmp_root, ignore_errors=True)
        (ws / "output").mkdir(parents=True)
        if docs_src.is_dir():
            shutil.copytree(docs_src, ws / "documents")
            for p in (ws / "documents").rglob("*"):
                if p.is_file():
                    p.chmod(0o444)
        else:
            (ws / "documents").mkdir()
        harness_dir = Path(args.harness_dir).resolve() if args.harness_dir else None
        installed = install_harness(
            ws, harness_dir, lab_root / "opencode" / "scripts" / "read_doc.py"
        )

        substrate = ClaudeSubstrate()
        argv = substrate.exec_argv(
            build_prompt(lab_root, args.task, ws), args.model, effort=args.reasoning_effort
        )
        (run_dir / "config.json").write_text(
            json.dumps(
                {
                    "runner": "claude",
                    "claude_bin": argv[0],
                    "model": args.model,
                    "reasoning_effort": args.reasoning_effort,
                    "task": args.task,
                    "run_id": args.run_id,
                    "timeout": args.timeout,
                    "harness_dir": str(harness_dir) if harness_dir else None,
                    "harness_files": installed,
                    "doc_parser": args.doc_parser,
                    "workspace": str(ws),
                    "started_at": started,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        env = {
            **os.environ,
            "LAB_DOC_PARSER": args.doc_parser,
            "LAB_BENCH_ROOT": str(lab_root),
        }
        env.pop("XDG_DATA_HOME", None)  # read_doc.py sandbox mode needs podman's store
        cmd = ["timeout", "--kill-after=30s", str(int(max(60, args.timeout))), *argv]
        session_path = run_dir / "claude_session.jsonl"
        with (
            open(session_path, "w", encoding="utf-8") as out,
            open(run_dir / "claude_stderr.log", "w", encoding="utf-8") as err,
        ):
            rc = subprocess.run(
                cmd, cwd=str(ws), env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err
            ).returncode
        wall = time.time() - t0

        graded = run_dir / "output"
        graded.mkdir(parents=True, exist_ok=True)
        for src in (ws / "output").rglob("*"):
            if src.is_file():
                dest = graded / src.relative_to(ws / "output")
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
        delivered = sorted(str(p.relative_to(graded)) for p in graded.rglob("*") if p.is_file())

        summary = parse_session(session_path)
        if summary["n_events"] == 0:
            print(
                f"[lab_claude_runner] NO parseable events in {session_path}; "
                f"NOT writing metrics.json (rc={rc})",
                file=sys.stderr,
            )
            return 2
        if rc == 0 and not delivered:
            rc = 1  # exit 0 with an empty output/ is a failure, not a success
        session_text = session_path.read_text(encoding="utf-8", errors="replace")
        docs = [str(p.relative_to(ws / "documents")) for p in (ws / "documents").rglob("*")]
        docs = [r for r in docs if (ws / "documents" / r).is_file()]
        scratch_left = sorted(
            str(p.relative_to(ws))
            for p in ws.rglob("*")
            if p.is_file()
            and not str(p.relative_to(ws)).startswith(("documents/", "output/"))
            and str(p.relative_to(ws)) not in installed
        )[:200]
        metrics = {
            "runner": "claude",
            "model": args.model,
            "task": args.task,
            "run_id": args.run_id,
            "turn_count": summary["turns"],
            "tool_calls": summary["tool_calls"],
            "stream_errors": summary["errors"],
            "input_tokens": summary["input_tokens"],
            "input_tokens_note": "input + cache creation + cache read, as reported by Claude Code.",
            "cached_input_tokens": summary["cached_input_tokens"],
            "output_tokens": summary["output_tokens"],
            "reasoning_tokens": summary["reasoning_output_tokens"],
            "total_tokens": summary["input_tokens"] + summary["output_tokens"],
            "cost_usd": summary["cost_usd"],
            "wall_clock_seconds": round(wall, 1),
            "finished_cleanly": rc == 0,
            "exit_code": rc,
            "deliverables": delivered,
            "documents_read": sum(1 for r in docs if r in session_text),
            "total_documents": len(docs),
            "harness_dir": str(harness_dir) if harness_dir else None,
            "harness_files": installed,
            "workspace_scratch_files": scratch_left,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(
            f"[lab_claude_runner] {args.run_id}: rc={rc} turns={summary['turns']} "
            f"tools={summary['tool_calls']} wall={wall:.0f}s "
            f"deliverables={', '.join(delivered) or 'NONE'}"
        )
        return rc
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


parser = argparse.ArgumentParser(description="Run one LAB task through Claude Code")
parser.add_argument("--task", required=True, help="Task ID (e.g. corporate-ma/<slug>)")
parser.add_argument("--run-id", required=True, help="Run identifier (namespaced)")
parser.add_argument("--results-root", required=True, help="Where <run-id>/ is written")
parser.add_argument("--lab-root", required=True, help="The harvey-labs checkout")
parser.add_argument("--scratch", required=True, help="Root for per-rollout workspaces")
parser.add_argument("--model", required=True, help="Claude model id")
parser.add_argument("--reasoning-effort", default="high", help="claude --effort")
parser.add_argument("--timeout", type=int, default=3600, help="Wall-clock budget in seconds")
parser.add_argument(
    "--harness-dir",
    default=None,
    help="Materialised claude_root tree (CLAUDE.md, .claude/, scripts/). Omit for bare.",
)
parser.add_argument(
    "--doc-parser",
    choices=["sandbox", "host"],
    default=os.environ.get("LAB_DOC_PARSER", "sandbox"),
    help="Where scripts/read_doc.py parses binary documents (default: %(default)s)",
)

if __name__ == "__main__":
    sys.exit(main(parser.parse_args()))
