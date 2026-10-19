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

"""ClaudeSubstrate: Claude Code (`claude -p`) as a solver substrate.

Carries the c1..c8 surface (CLAUDE_SURFACE), the one install mode (claude_root: CLAUDE.md,
.claude/ and scripts/ copied to the workspace root, where `claude -p` run from that directory
finds them), the `claude -p` flag contract, and the stream-json trajectory renderer.

Sub-agents keep the shared `sub_agents/<name>/AGENT.md` tree path the core validates; the
install step rewrites each into `.claude/agents/<name>.md`, the location Claude Code loads.
Runs are isolated from the operator's own Claude Code setup by a fresh CLAUDE_CONFIG_DIR and
`--setting-sources project`, so only the harness under test is in context.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Mapping

from .. import config
from ..core import runmark as RM
from ..core import tree as HT
from ..core.tree import SurfaceSpec
from .base import Install, Launcher, RunRequest

CLAUDE_SURFACE = SurfaceSpec(
    name="claude",
    always_on=("CLAUDE.md",),
    always_on_max=12_000,
    always_on_cap_reason=(
        "CLAUDE.md is loaded whole into the context of EVERY session, ahead of any work"
    ),
    skill_dir=".claude/skills",
    skill_name_rule="overrides_dir",
    trigger_hint=(
        'Fix it by adding a trigger line naming `{name}` to CLAUDE.md, e.g. "When <the '
        'situation this skill handles>, use the `{name}` skill."'
    ),
    autotrigger_file="CLAUDE.md",
    trigger_line="- When {desc}, use the `{name}` skill.\n",
    subagent_requires_mode=False,
    plugin_dir=None,
    hooks_file=".claude/settings.json",
    hook_events=frozenset(
        {
            "PreToolUse",
            "PostToolUse",
            "UserPromptSubmit",
            "Notification",
            "Stop",
            "SubagentStop",
            "PreCompact",
            "SessionStart",
            "SessionEnd",
        }
    ),
    compile_skill_scripts=True,
    reject_empty_always_on=True,
    stubs=(
        (
            "scripts/README.md",
            "<!-- c7: deterministic helpers, run as `python3 scripts/<name>.py`. "
            "Add one by emitting scripts/<name>.py. -->\n",
        ),
        (
            "sub_agents/README.md",
            "<!-- c5: delegated Claude Code subagents, each spawned in its OWN context. Add one "
            "as sub_agents/<name>/AGENT.md with frontmatter `description:`; the runner installs "
            "it as .claude/agents/<name>.md. -->\n",
        ),
        (
            ".claude/README.md",
            "<!-- c8: Claude Code lifecycle hooks -- the only lever that acts INSIDE the tool "
            "loop, so the model cannot skip it. Add ONE file, .claude/settings.json: "
            '{"hooks": {"PostToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", '
            '"command": "..."}]}]}}. A PostToolUse command that prints '
            '{"decision":"block","reason":"..."} feeds the reason back to the model. -->\n',
        ),
        (
            ".claude/skills/README.md",
            "<!-- c6: reusable workflows. Add one as .claude/skills/<name>/SKILL.md with YAML "
            "frontmatter: `description:` is REQUIRED; `name:` is optional and must match the "
            "directory. Only name+description are advertised to the agent; the body is read "
            "on demand. -->\n",
        ),
    ),
)


CLAUDE_ROOT_ENTRIES = ("CLAUDE.md", ".claude", "scripts")
_STUB_FILES = frozenset(rel for rel, _ in CLAUDE_SURFACE.stubs)


def agent_file(name: str, body: str) -> str:
    """An AGENT.md rewritten as a Claude Code subagent file: frontmatter `name:` is the id."""
    m = re.match(r"^---\n(.*?)\n---\n?", body, re.S)
    if not m:
        return f"---\nname: {name}\ndescription: {name}\n---\n{body}"
    front = [ln for ln in m.group(1).splitlines() if not re.match(r"^name:\s*", ln)]
    return "---\n" + "\n".join([f"name: {name}", *front]) + "\n---\n" + body[m.end() :]


class ClaudeSubstrate:
    """Claude Code behind the one Substrate interface."""

    name = "cc"

    def __init__(self, launcher: Launcher | None = None):
        self._launcher = launcher

    def surface(self) -> SurfaceSpec:
        return CLAUDE_SURFACE

    def materialise(self, tree_dir: Path, spec, ctx: Mapping[str, str]) -> Install:
        """Stage the claude_root entries for a copy to the workspace root, with every
        sub_agents/<name>/AGENT.md installed as .claude/agents/<name>.md; there is no render
        step, because the CLI loads CLAUDE.md, skills, agents and hooks itself."""
        if spec.install_mode != "claude_root":
            raise ValueError(
                f"install_mode {spec.install_mode!r} is not the claude mode; this substrate "
                f"has exactly one: 'claude_root'"
            )
        tree_dir = Path(tree_dir)
        if not HT.is_tree(CLAUDE_SURFACE, tree_dir):
            raise ValueError(f"{tree_dir} is not a claude harness tree (no CLAUDE.md)")
        staged = Path(tempfile.mkdtemp(prefix="adapt-cc-root-"))
        for entry in CLAUDE_ROOT_ENTRIES:
            src = tree_dir / entry
            if not src.exists():
                continue
            if src.is_dir():
                shutil.copytree(src, staged / entry)
            else:
                shutil.copy2(src, staged / entry)
        for rel in _STUB_FILES:
            (staged / rel).unlink(missing_ok=True)
        tree = HT.read_tree(CLAUDE_SURFACE, tree_dir)
        for name in HT.subagent_names(CLAUDE_SURFACE, tree):
            dst = staged / ".claude" / "agents" / f"{name}.md"
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(agent_file(name, tree[f"sub_agents/{name}/AGENT.md"]), encoding="utf-8")
        return Install(
            staging_dir=staged,
            fingerprint=RM.fingerprint(tree_dir),
            extras={
                "install_mode": "claude_root",
                **{k: v for k, v in ctx.items() if not k.startswith("file:")},
            },
        )

    def exec_argv(self, prompt: str, model: str, *, effort: str = "high") -> list[str]:
        """The `claude -p` argv, run with cwd set to the staged workspace.

        stream-json + --verbose keeps every tool call; --setting-sources project loads the
        harness's .claude/settings.json and nothing from the operator's machine. Callers use
        stdin=DEVNULL and `exec_env`.
        """
        return [
            config.CLAUDE_BIN,
            "-p",
            prompt,
            "--model",
            model,
            "--effort",
            effort,
            "--output-format",
            "stream-json",
            "--verbose",
            "--dangerously-skip-permissions",
            "--setting-sources",
            "project",
        ]

    def exec_env(self, base_env: Mapping[str, str], *, config_dir: Path) -> dict[str, str]:
        """A fresh CLAUDE_CONFIG_DIR, so user-level memory, skills, agents and plugins stay out.

        Auth then comes from the environment alone: ANTHROPIC_API_KEY, or Vertex when
        HARNESS2_CLAUDE_AUTH resolves to vertex.
        """
        env = dict(base_env)
        env.pop("CLAUDECODE", None)
        env["CLAUDE_CONFIG_DIR"] = str(config_dir)
        env.update(config.claude_auth_env())
        return env

    def execute(
        self, task_ref: str, install: Install, run_ids: list[str], timeout: int
    ) -> list[Path]:
        if self._launcher is None:
            raise RuntimeError(
                "ClaudeSubstrate was constructed without a launcher body; the benchmark "
                "adapter must supply one (it owns workspace layout and read-back locations)"
            )
        return self._launcher(
            RunRequest(task_ref=task_ref, run_ids=tuple(run_ids), timeout=int(timeout)), install
        )

    def read_trajectory(
        self, stream_path: Path, *, obs_trunc: int, max_chars: int, args_trunc: int = 2000
    ) -> str:
        return render_events(
            Path(stream_path), obs_trunc=obs_trunc, max_chars=max_chars, args_trunc=args_trunc
        )

    def preflight(self) -> None:
        """Check the CLI runs and that a non-interactive auth route is configured."""
        try:
            r = subprocess.run(
                [config.CLAUDE_BIN, "--version"], capture_output=True, text=True, timeout=30
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(
                f"Claude Code CLI not runnable ({type(exc).__name__}); install it with "
                f"`npm install -g @anthropic-ai/claude-code` or set HARNESS2_CLAUDE_BIN"
            ) from exc
        if r.returncode != 0:
            raise RuntimeError(f"`{config.CLAUDE_BIN} --version` failed: {r.stderr.strip()[-200:]}")
        if config.CLAUDE_AUTH == "vertex":
            if not config.VERTEX_PROJECT:
                raise config.ConfigError("HARNESS2_CLAUDE_AUTH=vertex needs GOOGLE_CLOUD_PROJECT")
        elif not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            raise config.ConfigError(
                "Claude Code rollouts run with an isolated config dir, so a stored login is not "
                "used: set ANTHROPIC_API_KEY, or HARNESS2_CLAUDE_AUTH=vertex with "
                "GOOGLE_CLOUD_PROJECT"
            )


_TRUNC_FMT = "…(+{n} chars — render truncation)"


def _clip(s, n: int) -> str:
    s = "" if s is None else str(s)
    s = s.replace("\x00", "")
    if n <= 0 or len(s) <= n:
        return s
    return s[:n] + _TRUNC_FMT.format(n=len(s) - n)


def _text_of(content) -> str:
    """Text of a message or tool_result `content`: a string, or a list of typed blocks."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, str):
                parts.append(b)
            elif isinstance(b, dict) and b.get("type") == "text":
                parts.append(str(b.get("text") or ""))
            elif isinstance(b, dict) and b.get("type") == "image":
                parts.append("(image)")
        return "\n".join(p for p in parts if p.strip())
    if isinstance(content, dict):
        return _text_of(content.get("content") or content.get("text"))
    return str(content)


def _tool_args(args) -> str:
    """A shell call renders as its command line, like the other substrates; anything else
    as its JSON input."""
    if isinstance(args, dict):
        if isinstance(args.get("command"), str):
            return args["command"]
        return json.dumps(args, ensure_ascii=False)
    return str(args)


def parse_events(path: Path) -> list[dict]:
    """Read the stream-json lines in order; non-JSON lines (stderr bleed) are tolerated."""
    try:
        raw = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return [{"type": "_read_error", "message": str(e)}]
    events = []
    for ln in raw.replace("\x00", "").splitlines():
        ln = ln.strip()
        if not ln.startswith("{"):
            continue
        try:
            ev = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if isinstance(ev, dict):
            events.append(ev)
    return events


def _render_message(ev: dict, lines: list[str], obs_trunc: int, args_trunc: int) -> None:
    msg = ev.get("message")
    if not isinstance(msg, dict):
        return
    sub = "SUBAGENT " if ev.get("parent_tool_use_id") else ""
    content = msg.get("content")
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    if not isinstance(content, list):
        return
    role = ev.get("type")
    for b in content:
        if not isinstance(b, dict):
            continue
        bt = b.get("type")
        if bt == "tool_use":
            name = b.get("name") or "tool"
            lines.append(f"[{sub}ACTION] {name}: {_clip(_tool_args(b.get('input')), args_trunc)}")
        elif bt == "tool_result":
            err = " ERROR" if b.get("is_error") else ""
            lines.append(f"[{sub}OBSERVATION{err}] {_clip(_text_of(b.get('content')), obs_trunc)}")
        elif bt == "thinking":
            txt = str(b.get("thinking") or "")
            if txt.strip():
                lines.append(f"[{sub}REASONING] {_clip(txt, obs_trunc)}")
        elif bt == "text":
            txt = str(b.get("text") or "")
            if txt.strip():
                tag = "ASSISTANT" if role == "assistant" else "USER"
                lines.append(f"[{sub}{tag}] {_clip(txt, obs_trunc)}")


def render_events(path: Path, *, obs_trunc: int, max_chars: int, args_trunc: int = 2000) -> str:
    """Render a `claude -p --output-format stream-json --verbose` stream into the shared
    grammar. `obs_trunc` and `max_chars` come from the cell's BenchSpec; `args_trunc` is a
    per-substrate default that BenchSpec does not carry."""
    lines: list[str] = []
    for ev in parse_events(Path(path)):
        et = ev.get("type", "")
        if et == "_read_error":
            return f"(failed to read trajectory: {ev.get('message')})"
        if et in ("assistant", "user"):
            _render_message(ev, lines, obs_trunc, args_trunc)
        elif et == "system" and ev.get("subtype") == "init":
            lines.append(
                f"[HARNESS] session start | model={ev.get('model')} | "
                f"agents={','.join(map(str, ev.get('agents') or []))}"
            )
        elif et == "system" and ev.get("subtype") in ("compact_boundary", "api_error"):
            lines.append(f"[HARNESS] {ev.get('subtype')}")
        elif et == "result":
            usage = ev.get("usage") if isinstance(ev.get("usage"), dict) else {}
            lines.append(
                "[HARNESS] session {st} | turns={t} cost_usd={c} | tokens in={i} "
                "cache_read={cr} out={o}".format(
                    st=ev.get("subtype"),
                    t=ev.get("num_turns"),
                    c=ev.get("total_cost_usd"),
                    i=usage.get("input_tokens"),
                    cr=usage.get("cache_read_input_tokens"),
                    o=usage.get("output_tokens"),
                )
            )
            if ev.get("is_error"):
                lines.append(f"[ERROR] {_clip(ev.get('result') or ev.get('subtype'), 1000)}")
    rendered = "\n".join(lines).strip()
    if not rendered:
        return "(no parseable events in trajectory)"
    if len(rendered) > max_chars:
        dropped = len(rendered) - max_chars
        rendered = rendered[:max_chars] + "\n" + _TRUNC_FMT.format(n=dropped)
    return rendered


def final_message(path: Path) -> str:
    """The `result` text of the last result event, else the last top-level assistant text."""
    last = ""
    for ev in parse_events(Path(path)):
        if ev.get("type") == "result" and isinstance(ev.get("result"), str):
            last = ev["result"] or last
        elif ev.get("type") == "assistant" and not ev.get("parent_tool_use_id"):
            msg = ev.get("message")
            txt = _text_of(msg.get("content")) if isinstance(msg, dict) else ""
            if txt.strip():
                last = txt
    return last
