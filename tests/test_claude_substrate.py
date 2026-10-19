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

"""The Claude Code substrate: surface, install, trajectory rendering and the JobBench cell."""

import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from harness2 import config
from harness2.benchmarks import jobbench
from harness2.cli._common import make_adapter
from harness2.core import prompts
from harness2.core import tree as HT
from harness2.substrate.claude import (
    CLAUDE_SURFACE,
    ClaudeSubstrate,
    agent_file,
    final_message,
    render_events,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "claude_stream.jsonl"
RUNNER = Path(jobbench.__file__).resolve().parent / "jb_claude_runner.py"

GOOD_TREE = {
    "CLAUDE.md": "Do the task.\n- When verifying a workbook, use the `xlsx-check` skill.\n",
    ".claude/skills/xlsx-check/SKILL.md": (
        "---\ndescription: verify a written xlsx\n---\nRun the checker.\n"
        "python3 .claude/skills/xlsx-check/scripts/check.py out.xlsx\n"
    ),
    ".claude/skills/xlsx-check/scripts/check.py": "print('ok')\n",
    "sub_agents/doc-reader/AGENT.md": "---\ndescription: read many docs, return a table\n---\nRead.\n",
    "scripts/tidy.py": "print('tidy')\n",
    ".claude/settings.json": json.dumps(
        {"hooks": {"Stop": [{"hooks": [{"type": "command", "command": "true"}]}]}}
    ),
}


class ClaudeSurfaceTests(unittest.TestCase):
    def test_surface_validates_and_rejects(self):
        census = HT.validate(CLAUDE_SURFACE, dict(GOOD_TREE))
        self.assertEqual((census["skills"], census["sub_agents"], census["hooks"]), (1, 1, 1))
        self.assertFalse(HT.is_allowed(CLAUDE_SURFACE, ".claude/agents/doc-reader.md"))
        self.assertFalse(HT.is_allowed(CLAUDE_SURFACE, "AGENTS.md"))
        untriggered = dict(GOOD_TREE, **{"CLAUDE.md": "Do the task.\n"})
        with self.assertRaisesRegex(ValueError, "never named in the always-on text"):
            HT.validate(CLAUDE_SURFACE, untriggered)
        bad_hook = dict(GOOD_TREE, **{".claude/settings.json": '{"hooks": {"OnExit": []}}'})
        with self.assertRaisesRegex(ValueError, "unknown event"):
            HT.validate(CLAUDE_SURFACE, bad_hook)

    def test_materialise_installs_subagents_where_claude_code_loads_them(self):
        with tempfile.TemporaryDirectory() as d:
            src = HT.write_tree(CLAUDE_SURFACE, GOOD_TREE, Path(d) / "tree")
            spec = SimpleNamespace(install_mode="claude_root")
            install = ClaudeSubstrate().materialise(src, spec, {"file:x": "y", "k": "v"})
            staged = install.staging_dir
            try:
                agent = (staged / ".claude/agents/doc-reader.md").read_text()
                self.assertTrue(agent.startswith("---\nname: doc-reader\ndescription:"))
                self.assertFalse((staged / "sub_agents").exists())
                self.assertFalse((staged / ".claude/README.md").exists())
                self.assertTrue((staged / ".claude/skills/xlsx-check/scripts/check.py").is_file())
                self.assertTrue((staged / "scripts/tidy.py").is_file())
                self.assertEqual(install.extras, {"install_mode": "claude_root", "k": "v"})
            finally:
                subprocess.run(["rm", "-rf", str(staged)], check=True)
            with self.assertRaises(ValueError):
                ClaudeSubstrate().materialise(src, SimpleNamespace(install_mode="codex_root"), {})

    def test_agent_file_overrides_name_and_adds_missing_frontmatter(self):
        self.assertEqual(
            agent_file("a", "---\nname: other\ndescription: d\n---\nbody"),
            "---\nname: a\ndescription: d\n---\nbody",
        )
        self.assertTrue(agent_file("a", "plain body").startswith("---\nname: a\n"))

    def test_prompt_surface_names_claude_paths_only(self):
        st = prompts._st(CLAUDE_SURFACE)
        for text in (st.surface_md, st.tts_surface, st.inv, st.dom):
            self.assertNotIn("AGENTS.md", text)
            self.assertNotIn(".codex", text)
        self.assertIn("`.claude/settings.json`", st.tts_surface)
        self.assertIn("`CLAUDE.md`", st.surface_md)


class ClaudeTrajectoryTests(unittest.TestCase):
    def test_render_events_shared_grammar(self):
        out = render_events(FIXTURE, obs_trunc=50, max_chars=100_000)
        self.assertIn("[HARNESS] session start | model=claude-sonnet-5-5", out)
        self.assertIn("[REASONING] I should read the instructions first.", out)
        self.assertIn("[ACTION] Bash: cat task_folder/TASK_INSTRUCTIONS.txt", out)
        self.assertIn("[OBSERVATION] Build a budget in output/budget.xlsx", out)
        self.assertIn('[ACTION] Agent: {"subagent_type": "doc-reader"', out)
        self.assertIn("[SUBAGENT ACTION] Read:", out)
        self.assertIn("[SUBAGENT OBSERVATION] item,cost", out)
        self.assertIn("[OBSERVATION ERROR] Exit code 1", out)
        self.assertIn("[ASSISTANT] Wrote output/budget.xlsx.", out)
        self.assertIn("[HARNESS] session success | turns=5 cost_usd=0.0123", out)
        self.assertNotIn("rate_limit", out)
        self.assertNotIn("stderr bleed", out)

    def test_render_truncation_and_missing_file(self):
        clipped = render_events(FIXTURE, obs_trunc=10, max_chars=200)
        self.assertIn("render truncation", clipped)
        self.assertLessEqual(len(clipped.split("\n…")[0]), 200)
        self.assertTrue(
            render_events(Path("/nonexistent.jsonl"), obs_trunc=10, max_chars=10).startswith(
                "(failed to read trajectory"
            )
        )

    def test_final_message(self):
        self.assertEqual(final_message(FIXTURE), "Wrote output/budget.xlsx.")


class ClaudeCellTests(unittest.TestCase):
    def test_jobbench_cell_and_cli_scope(self):
        adapter = jobbench.JobBenchAdapter(substrate="cc", store_root=Path(tempfile.mkdtemp()))
        self.assertEqual(adapter.substrate.name, "cc")
        self.assertEqual(adapter.spec.install_mode, "claude_root")
        self.assertTrue(adapter.spec.allow_subagents)
        base = adapter.base_harness()
        self.assertTrue(HT.is_tree(CLAUDE_SURFACE, base))
        HT.validate(CLAUDE_SURFACE, HT.read_tree(CLAUDE_SURFACE, base))
        self.assertEqual(make_adapter("jb", None, "cc").substrate.name, "cc")
        with self.assertRaisesRegex(ValueError, "wired for jb, lab only"):
            make_adapter("wb", "office", "cc")

    def test_validate_models_for_claude_executor(self):
        with (
            patch.multiple(
                config,
                IMPROVER_MODEL="claude-opus-5-5",
                DEFAULT_PROPOSER_CHANNEL="claude_cli",
                CLAUDE_AUTH="anthropic",
            ),
            patch.dict(os.environ, {"HARNESS2_IMPROVER_MODEL": "claude-opus-5-5"}),
        ):
            config.validate_models(substrate="cc")
            with patch.object(config, "DEFAULT_PROPOSER_CHANNEL", "opencode"):
                with self.assertRaisesRegex(config.ConfigError, "does not fit improver channel"):
                    config.validate_models(substrate="cc")

    def test_exec_env_isolates_config_and_routes_auth(self):
        base = {"CLAUDECODE": "1", "PATH": "/bin", "ANTHROPIC_API_KEY": "k"}
        with patch.object(config, "CLAUDE_AUTH", "anthropic"):
            env = ClaudeSubstrate().exec_env(base, config_dir=Path("/tmp/cfg"))
        self.assertEqual(env["CLAUDE_CONFIG_DIR"], "/tmp/cfg")
        self.assertNotIn("CLAUDECODE", env)
        self.assertEqual(env["ANTHROPIC_API_KEY"], "k")
        with patch.multiple(config, CLAUDE_AUTH="vertex", VERTEX_PROJECT="p"):
            env = ClaudeSubstrate().exec_env(base, config_dir=Path("/tmp/cfg"))
        self.assertEqual(env["CLAUDE_CODE_USE_VERTEX"], "1")
        self.assertEqual(env["ANTHROPIC_VERTEX_PROJECT_ID"], "p")
        argv = ClaudeSubstrate().exec_argv("do it", "claude-sonnet-5-5")
        for flag in ("-p", "stream-json", "--verbose", "--setting-sources", "project"):
            self.assertIn(flag, argv)


FAKE_CLAUDE = """#!/usr/bin/env python3
import json, os, pathlib, sys
cwd = pathlib.Path.cwd()
assert (cwd / "CLAUDE.md").is_file(), "harness not staged at the workspace root"
assert (cwd / ".claude/agents/doc-reader.md").is_file(), "subagent not installed"
assert os.environ.get("CLAUDE_CONFIG_DIR"), "config dir not isolated"
assert sys.argv[sys.argv.index("--model") + 1] == "claude-test"
(cwd / "output" / "answer.md").write_text("done")
print(json.dumps({"type": "assistant", "parent_tool_use_id": None,
                  "message": {"content": [{"type": "text", "text": "done"}]}}))
print(json.dumps({"type": "result", "subtype": "success", "result": "done"}))
"""


class ClaudeRunnerTests(unittest.TestCase):
    def test_runner_stages_runs_and_moves_deliverables(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            fake = d / "claude"
            fake.write_text(FAKE_CLAUDE)
            fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
            task = d / "store" / "accountant" / "task1"
            (task / "task_folder").mkdir(parents=True)
            (task / "task_folder" / "TASK_INSTRUCTIONS.txt").write_text("write answer.md")
            src = HT.write_tree(CLAUDE_SURFACE, GOOD_TREE, d / "tree")
            install = ClaudeSubstrate().materialise(
                src, SimpleNamespace(install_mode="claude_root"), {}
            )
            env = {
                **os.environ,
                "HARNESS2_CLAUDE_BIN": str(fake),
                "CLAUDE_CONFIG_DIR": str(d / "cfg"),
                "TASK_IDS": "accountant/task1",
                "LABEL": "jb-cc-agg1__base",
                "HARNESS_RENDER_DIR": str(install.staging_dir),
                "TASKS_BASE_DIR": str(d / "store"),
                "MODEL_ID": "claude-test",
                "TIMEOUT_PER_TASK": "60",
                "MAX_RETRIES": "1",
            }
            r = subprocess.run(
                [sys.executable, str(RUNNER)], env=env, capture_output=True, text=True, timeout=120
            )
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            out = task / "model_output" / "jb-cc-agg1__base" / "answer.md"
            self.assertEqual(out.read_text(), "done")
            traj = list((task / "model_traj" / "jb-cc-agg1__base").glob("*.jsonl"))
            self.assertEqual(len(traj), 1)
            self.assertIn("[ASSISTANT] done", render_events(traj[0], obs_trunc=100, max_chars=1000))
            subprocess.run(["rm", "-rf", str(install.staging_dir)], check=True)


if __name__ == "__main__":
    unittest.main()
