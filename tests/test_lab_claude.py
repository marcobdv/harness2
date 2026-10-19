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

"""The LAB cell on the Claude Code substrate, against a fake harvey-labs checkout."""

import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from harness2 import config
from harness2.benchmarks import lab
from harness2.benchmarks.lab_claude_runner import parse_session
from harness2.core import runmark as RM
from harness2.core import tree as HT
from harness2.substrate.claude import CLAUDE_SURFACE

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "claude_stream.jsonl"
TASK = "corporate-ma/nda-review"

FAKE_CLAUDE = """#!/usr/bin/env python3
import json, os, pathlib, sys
ws = pathlib.Path.cwd()
assert "Treat it as" in (ws / "CLAUDE.md").read_text(), "base harness not installed"
assert (ws / "scripts/read_doc.py").is_file(), "runtime read_doc.py not installed"
assert not (ws / ".claude/README.md").exists(), "base tree was not materialised"
assert os.environ.get("CLAUDE_CONFIG_DIR"), "config dir not isolated"
assert os.environ.get("LAB_BENCH_ROOT"), "LAB env missing"
prompt = sys.argv[sys.argv.index("-p") + 1]
assert "Review the NDA" in prompt and "criteria" not in prompt, prompt
(ws / "output" / "memo.md").write_text("NDA memo")
for ev in (
    {"type": "assistant", "parent_tool_use_id": None, "message": {"content": [
        {"type": "tool_use", "id": "t1", "name": "Bash",
         "input": {"command": "python3 scripts/read_doc.py documents/nda.docx"}}]}},
    {"type": "user", "parent_tool_use_id": None, "message": {"content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": "NDA text"}]}},
    {"type": "result", "subtype": "success", "is_error": False, "num_turns": 2,
     "result": "done", "total_cost_usd": 0.5,
     "usage": {"input_tokens": 10, "cache_read_input_tokens": 90, "output_tokens": 7}},
):
    print(json.dumps(ev))
"""


class LabClaudeTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        repo = self.root / "harvey-labs"
        task_dir = repo / "tasks" / TASK
        (task_dir / "documents").mkdir(parents=True)
        (task_dir / "documents" / "nda.docx").write_text("binary")
        (task_dir / "task.json").write_text(
            json.dumps(
                {
                    "title": "NDA review",
                    "instructions": "Review the NDA and write output/memo.md.",
                    "deliverables": {"memo.md": "memo"},
                    "criteria": ["secret rubric"],
                }
            )
        )
        (repo / "opencode" / "scripts").mkdir(parents=True)
        (repo / "opencode" / "scripts" / "read_doc.py").write_text("print('doc')\n")
        fake = self.root / "claude"
        fake.write_text(FAKE_CLAUDE)
        fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
        for p in (
            patch.object(config, "RUNS", self.root / "runs"),
            patch.object(config, "lab_repo", lambda: repo),
            patch.dict(os.environ, {"HARNESS2_CLAUDE_BIN": str(fake)}),
            patch("harness2.substrate.claude.ClaudeSubstrate.preflight", lambda self: None),
        ):
            p.start()
            self.addCleanup(p.stop)
        self.adapter = lab.LabAdapter(substrate="cc")

    def test_spec_base_harness_and_invariants(self):
        self.assertEqual(self.adapter.spec.install_mode, "claude_root")
        self.assertTrue(self.adapter.spec.allow_subagents)
        base = self.adapter.base_harness()
        tree = HT.read_tree(CLAUDE_SURFACE, base)
        self.assertEqual(tree["CLAUDE.md"], lab.CODEX_BASE_BODY)
        self.adapter.validate_tree(tree)
        with self.assertRaisesRegex(ValueError, "CLAUDE.md must say deliverables go to"):
            self.adapter.validate_tree({"CLAUDE.md": "read via scripts/read_doc.py"})
        with self.assertRaisesRegex(ValueError, "CLAUDE.md must say .docx"):
            self.adapter.validate_tree({"CLAUDE.md": "write to output/x"})

    def test_execute_base_rollout_end_to_end(self):
        out = self.adapter.execute(TASK, None, "lab-cc-agg1/base/r0", timeout=120)
        self.assertEqual((out / "output" / "memo.md").read_text(), "NDA memo")
        metrics = json.loads((out / "metrics.json").read_text())
        self.assertEqual(metrics["runner"], "claude")
        self.assertEqual(metrics["deliverables"], ["memo.md"])
        self.assertEqual((metrics["documents_read"], metrics["total_documents"]), (1, 1))
        self.assertEqual((metrics["input_tokens"], metrics["cached_input_tokens"]), (100, 90))
        self.assertTrue(metrics["finished_cleanly"])
        self.assertTrue(RM.is_complete(out, harness_dir=None))
        traj = self.adapter.trajectory(out, obs_trunc=1000)
        self.assertIn("[ACTION] Bash: python3 scripts/read_doc.py documents/nda.docx", traj)
        self.assertIn("[OBSERVATION] NDA text", traj)
        self.assertTrue(self.adapter.delivered(out))
        self.assertTrue(self.adapter.artifacts_intact(out))

    def test_parse_session_summary(self):
        s = parse_session(FIXTURE)
        self.assertEqual(s["tool_calls"], 4)
        self.assertEqual(
            s["commands"],
            [
                "cat task_folder/TASK_INSTRUCTIONS.txt",
                "python3 scripts/make_xlsx.py",
            ],
        )
        self.assertEqual((s["turns"], s["errors"], s["cost_usd"]), (5, 0, 0.0123))
        self.assertEqual(s["input_tokens"], 4012)


if __name__ == "__main__":
    unittest.main()
