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

"""JobBench Claude Code launcher: stage a task workspace, run `claude -p` in it, keep the
artifacts.

The environment contract mirrors jb_opencode_runner.py. The adapter has already built the
process environment (isolated CLAUDE_CONFIG_DIR, auth routing) via ClaudeSubstrate.exec_env;
this module only stages files, runs the CLI and moves deliverables.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from harness2.benchmarks.jb_opencode_runner import PROMPT
from harness2.substrate.claude import CLAUDE_ROOT_ENTRIES, ClaudeSubstrate

TEMP_BASE = Path("/tmp/hcc_ws")
TRAJ_DIR_NAME = "model_traj"


@dataclass(frozen=True)
class Settings:
    task_ids: tuple[str, ...]
    label: str
    render_dir: Path
    tasks_base: Path
    model_id: str
    effort: str
    timeout_per_task: str
    max_retries: int
    log_dir: Path


def _settings() -> Settings:
    """Resolve the environment contract the adapter writes; a bad one exits non-zero."""
    env = os.environ
    missing = [n for n in ("MODEL_ID", "LABEL", "HARNESS_RENDER_DIR", "TASK_IDS") if not env.get(n)]
    if missing:
        raise SystemExit(f"[ERROR] unset: {', '.join(missing)}")
    render_dir = Path(env["HARNESS_RENDER_DIR"])
    if not (render_dir / "CLAUDE.md").is_file():
        raise SystemExit(f"[ERROR] HARNESS_RENDER_DIR has no CLAUDE.md: {render_dir}")
    if not env.get("CLAUDE_CONFIG_DIR"):
        raise SystemExit(
            "[ERROR] CLAUDE_CONFIG_DIR unset; the rollout would load the operator's setup"
        )
    tasks_base = Path(env["TASKS_BASE_DIR"])
    return Settings(
        task_ids=tuple(env["TASK_IDS"].split()),
        label=env["LABEL"],
        render_dir=render_dir,
        tasks_base=tasks_base,
        model_id=env["MODEL_ID"],
        effort=env.get("EFFORT") or "high",
        timeout_per_task=env.get("TIMEOUT_PER_TASK") or "3600",
        max_retries=int(env.get("MAX_RETRIES") or 2),
        log_dir=tasks_base / ".harness2-logs",
    )


def _hms() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _nonempty(path: Path) -> bool:
    return path.is_dir() and any(path.iterdir())


def _log(log_file: Path, line: str) -> None:
    print(line, flush=True)
    with log_file.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _reset_task(task_dir: Path, workspace: Path) -> None:
    """Restore a pristine task_folder and an empty output/ before a retry."""
    shutil.rmtree(workspace / "task_folder", ignore_errors=True)
    shutil.rmtree(workspace / "output", ignore_errors=True)
    shutil.copytree(task_dir, workspace / "task_folder")
    (workspace / "output").mkdir(parents=True, exist_ok=True)


def _stage(task_dir: Path, render_dir: Path, workspace: Path) -> None:
    """Task files under task_folder/, the harness at the workspace root where `claude -p`
    run from that directory loads it."""
    shutil.rmtree(workspace, ignore_errors=True)
    (workspace / "output").mkdir(parents=True)
    shutil.copytree(task_dir, workspace / "task_folder")
    for entry in CLAUDE_ROOT_ENTRIES:
        src = render_dir / entry
        if src.is_dir():
            shutil.copytree(src, workspace / entry)
        elif src.is_file():
            shutil.copy2(src, workspace / entry)


def _attempt(cfg: Settings, workspace: Path, traj_file: Path, log_file: Path) -> int:
    """Run Claude Code once; its stream-json lands in `traj_file`, stderr in the log."""
    prompt = PROMPT.format(
        task_folder=workspace / "task_folder",
        output_dir=workspace / "output",
        workspace=workspace,
    )
    cmd = ["timeout", "--kill-after=30s", cfg.timeout_per_task] + ClaudeSubstrate().exec_argv(
        prompt, cfg.model_id, effort=cfg.effort
    )
    handle, raw = tempfile.mkstemp(prefix=f".{traj_file.stem}.stdout.", dir=traj_file.parent)
    os.close(handle)
    with open(raw, "w", encoding="utf-8") as out, log_file.open("a", encoding="utf-8") as err:
        code = subprocess.run(
            cmd, cwd=workspace, stdin=subprocess.DEVNULL, stdout=out, stderr=err
        ).returncode
    with open(raw, encoding="utf-8", errors="replace") as src:
        traj_file.write_text("".join(ln for ln in src if ln.startswith("{")), encoding="utf-8")
    os.unlink(raw)
    return code


def run_one(cfg: Settings, task_id: str) -> bool:
    """Roll one task. True only when deliverables landed in `model_output/<label>/`; a
    False makes `main` exit non-zero, so the adapter never marks an output-less attempt
    complete and scores it as a real zero."""
    task_parent = cfg.tasks_base / task_id
    task_dir = task_parent / "task_folder"
    task_name = task_id.replace("/", "_")
    final_output_dir = task_parent / "model_output" / cfg.label
    traj_dir = task_parent / TRAJ_DIR_NAME / cfg.label
    log_file = cfg.log_dir / f"hcc_{task_name}_{cfg.label}_{_stamp()}.log"

    if not task_dir.is_dir():
        _log(log_file, f"[ERROR] no task_folder: {task_dir}")
        return False
    if _nonempty(final_output_dir):
        print(f"[{_hms()}] SKIP (done): {task_id} ({cfg.label})", flush=True)
        return True
    traj_dir.mkdir(parents=True, exist_ok=True)

    workspace = TEMP_BASE / f"{task_name}_{cfg.label}_{os.getpid()}"
    output_dir = workspace / "output"
    _stage(task_dir, cfg.render_dir, workspace)

    for attempt in range(1, cfg.max_retries + 1):
        if attempt > 1:
            _reset_task(task_dir, workspace)
        traj_file = traj_dir / f"{task_name}_{cfg.label}_attempt{attempt}_{_stamp()}.jsonl"
        _log(log_file, f"[{_hms()}] run {task_id} ({cfg.label}) attempt {attempt} -> {traj_file}")
        code = _attempt(cfg, workspace, traj_file, log_file)
        if _nonempty(output_dir):
            break
        if code in (124, 137):
            _log(log_file, f"[{_hms()}] TIMEOUT {task_id} attempt {attempt}")
        else:
            _log(log_file, f"[{_hms()}] no output {task_id} attempt {attempt} (exit {code})")

    delivered = _nonempty(output_dir)
    if delivered:
        final_output_dir.mkdir(parents=True, exist_ok=True)
        for item in sorted(output_dir.iterdir()):
            if not item.name.startswith("."):
                shutil.move(str(item), str(final_output_dir / item.name))
        _log(log_file, f"[{_hms()}] output -> {final_output_dir}")
    else:
        _log(log_file, f"[{_hms()}] NO OUTPUT for {task_id} ({cfg.label})")
    shutil.rmtree(workspace, ignore_errors=True)
    return delivered and _nonempty(final_output_dir)


def main() -> int:
    cfg = _settings()
    print(f"=== claude code eval: LABEL={cfg.label} MODEL={cfg.model_id} ===", flush=True)
    print(f"render: {cfg.render_dir}", flush=True)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    TEMP_BASE.mkdir(parents=True, exist_ok=True)
    failed = [task_id for task_id in cfg.task_ids if not run_one(cfg, task_id)]
    print(f"=== claude code eval done: {cfg.label} ===", flush=True)
    if failed:
        print(f"no output for: {', '.join(failed)}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
