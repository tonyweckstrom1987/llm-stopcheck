import contextlib
import io
import json
import os
import tempfile
import unittest

from llm_stopcheck import cli


def run(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(argv)
    return code, out.getvalue(), err.getvalue()


class CliTests(unittest.TestCase):
    def write(self, data):
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data if isinstance(data, str) else json.dumps(data))
        self.addCleanup(os.remove, path)
        return path

    def test_ok_exit_0(self):
        path = self.write({"choices": [{"finish_reason": "stop", "message": {"content": "hi"}}]})
        code, out, _ = run([path])
        self.assertEqual(code, 0)
        self.assertIn("stopcheck=ok", out)

    def test_truncated_exit_1_json(self):
        path = self.write({"choices": [{"finish_reason": "length", "message": {"content": "x"}}]})
        code, out, _ = run([path, "--json", "--max-tokens", "10"])
        self.assertEqual(code, 1)
        data = json.loads(out)
        self.assertEqual(data["code"], "truncated")
        self.assertEqual(data["max_tokens"], 10)

    def test_unreadable_exit_2(self):
        self.assertEqual(run(["/nonexistent/file.json"])[0], 2)
        self.assertEqual(run([self.write("not json")])[0], 2)

    def test_read_error_shows_only_type(self):
        d = tempfile.mkdtemp()
        self.addCleanup(os.rmdir, d)
        old = os.getcwd()
        os.chdir(d)
        self.addCleanup(os.chdir, old)
        code, _, err = run(["missing.json"])
        self.assertEqual(code, 2)
        self.assertEqual(err, "llm-stopcheck: cannot read input missing.json: FileNotFoundError\n")
        self.assertNotIn(d, err)
        self.assertNotIn("No such file", err)
        bad = self.write("not json")
        code, _, err = run([bad])
        self.assertEqual(code, 2)
        self.assertTrue(err.rstrip().endswith("JSONDecodeError"))
        self.assertNotIn("Expecting", err)

    def test_usage_exit_2(self):
        self.assertEqual(run([])[0], 2)


class CliExtraTests(CliTests):
    def test_reads_stdin_with_dash(self):
        import unittest.mock as mock
        data = json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": "hi"}}]})
        with mock.patch("sys.stdin", io.StringIO(data)):
            code, out, _ = run(["-"])
        self.assertEqual((code, "stopcheck=ok" in out), (0, True))

    def test_warnings_go_to_stderr_and_exit_stays_0(self):
        path = self.write({"choices": [{"finish_reason": "stop", "message": {"content": "hi"}}],
                           "usage": {"completion_tokens": 10}})
        code, out, err = run([path, "--max-tokens", "10"])
        self.assertEqual(code, 0)
        self.assertIn("warn=at_limit", out)
        self.assertIn("warnings: at_limit", err)

    def test_help_exits_0(self):
        code, out, _ = run(["--help"])
        self.assertEqual(code, 0)
        self.assertIn("usage:", out)

    def test_version(self):
        import llm_stopcheck
        code, out, _ = run(["--version"])
        self.assertEqual(code, 0)
        self.assertIn(llm_stopcheck.__version__, out)

    def test_near_limit_option(self):
        path = self.write({"choices": [{"finish_reason": "stop", "message": {"content": "hi"}}],
                           "usage": {"completion_tokens": 90}})
        code, out, _ = run([path, "--max-tokens", "100"])
        self.assertNotIn("near_limit", out)
        code, out, err = run([path, "--max-tokens", "100", "--near-limit", "0.9"])
        self.assertEqual(code, 0)
        self.assertIn("warnings: near_limit", err)

    def test_near_limit_out_of_range_exit_2(self):
        path = self.write({"choices": [{"finish_reason": "stop", "message": {"content": "hi"}}]})
        self.assertEqual(run([path, "--near-limit", "0"])[0], 2)
        self.assertEqual(run([path, "--near-limit", "1.5"])[0], 2)

    def test_allow_empty(self):
        path = self.write({"choices": [{"finish_reason": "stop", "message": {"content": ""}}]})
        code, out, _ = run([path])
        self.assertEqual(code, 1)
        self.assertIn("stopcheck=empty_output", out)
        code, out, _ = run([path, "--allow-empty"])
        self.assertEqual(code, 0)
        self.assertIn("stopcheck=ok", out)

    def test_multiple_files_worst_exit_code(self):
        good = self.write({"choices": [{"finish_reason": "stop", "message": {"content": "hi"}}]})
        bad = self.write({"choices": [{"finish_reason": "length", "message": {"content": "x"}}]})
        code, out, _ = run([good, bad, good])
        self.assertEqual(code, 1)
        self.assertEqual(len(out.splitlines()), 3)
        self.assertIn("stopcheck=truncated", out.splitlines()[1])
        self.assertEqual(run([good, good])[0], 0)

    def test_multiple_files_unreadable_gives_2_but_checks_the_rest(self):
        bad = self.write({"choices": [{"finish_reason": "length", "message": {"content": "x"}}]})
        code, out, err = run(["/nonexistent/file.json", bad])
        self.assertEqual(code, 2)
        self.assertIn("stopcheck=truncated", out)
        self.assertIn("/nonexistent/file.json", err)

    def test_multiple_files_output_names_the_file(self):
        good = self.write({"choices": [{"finish_reason": "stop", "message": {"content": "hi"}}]})
        bad = self.write({"choices": [{"finish_reason": "length", "message": {"content": "x"}}]})
        _, out, _ = run([good, bad])
        lines = out.splitlines()
        self.assertIn("stopcheck=ok", lines[0])
        self.assertIn(" file=", lines[0])
        self.assertIn(os.path.basename(good), lines[0])
        self.assertIn("stopcheck=truncated", lines[1])
        self.assertIn(os.path.basename(bad), lines[1])
        _, out, _ = run([good, bad, "--json"])
        records = [json.loads(line) for line in out.splitlines()]
        self.assertEqual([r["file"] for r in records], [good, bad])

    def test_multiple_files_long_or_spaced_path_stays_one_token(self):
        import shutil
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d)
        sub = os.path.join(d, "a" * 70, "my dir")
        os.makedirs(sub)
        paths = []
        for name, reason in (("first file.json", "stop"), ("second.json", "length")):
            path = os.path.join(sub, name)
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"choices": [{"finish_reason": reason, "message": {"content": "hi"}}]}, f)
            paths.append(path)
        code, out, _ = run(paths)
        self.assertEqual(code, 1)
        lines = out.splitlines()
        self.assertEqual(len(lines), 2)
        for line, path in zip(lines, paths):
            tokens = line.split(" ")
            self.assertTrue(all("=" in t for t in tokens), line)
            self.assertEqual(tokens[-1], "file=" + path.replace(" ", "_"))
        self.assertTrue(lines[0].endswith("/my_dir/first_file.json"))
        records = [json.loads(line) for line in run(paths + ["--json"])[1].splitlines()]
        self.assertEqual([r["file"] for r in records], paths)

    def test_bug_in_multi_file_output_is_reported_not_hidden(self):
        import unittest.mock as mock
        good = self.write({"choices": [{"finish_reason": "stop", "message": {"content": "hi"}}]})
        with mock.patch("llm_stopcheck.errors.format_record", side_effect=TypeError("boom")):
            code, out, err = run([good, good])
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertEqual(err.count("internal error on %s: TypeError" % good), 2)
        self.assertNotIn("boom", err)

    def test_single_file_output_has_no_file_name(self):
        good = self.write({"choices": [{"finish_reason": "stop", "message": {"content": "hi"}}]})
        self.assertNotIn("file=", run([good])[1])
        self.assertNotIn("file", json.loads(run([good, "--json"])[1]))

    def test_unexpected_exception_is_exit_2_with_type_name_only(self):
        import unittest.mock as mock
        good = self.write({"choices": [{"finish_reason": "stop", "message": {"content": "hi"}}]})
        with mock.patch("llm_stopcheck.core.check", side_effect=RuntimeError("secret-token-123")):
            code, out, err = run([good])
        self.assertEqual(code, 2)
        self.assertIn("RuntimeError", err)
        self.assertNotIn("secret-token-123", err + out)

    def test_exit_codes_are_distinct(self):
        good = self.write({"choices": [{"finish_reason": "stop", "message": {"content": "hi"}}]})
        bad = self.write({"choices": [{"finish_reason": "length", "message": {"content": "x"}}]})
        self.assertEqual([run([good])[0], run([bad])[0], run(["/nonexistent/x.json"])[0]], [0, 1, 2])

    def test_module_entry_point(self):
        import subprocess
        import sys
        src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
        path = self.write({"choices": [{"finish_reason": "length", "message": {"content": "x"}}]})
        r = subprocess.run([sys.executable, "-m", "llm_stopcheck", path], capture_output=True, text=True,
                           env=dict(os.environ, PYTHONPATH=src))
        self.assertEqual(r.returncode, 1)
        self.assertIn("stopcheck=truncated", r.stdout)

    def test_non_positive_max_tokens_is_usage_error(self):
        path = self.write({"choices": [{"finish_reason": "stop", "message": {"content": "hi"}}]})
        for value in ("0", "-5"):
            code, _, err = run([path, "--max-tokens", value])
            self.assertEqual(code, 2)
            self.assertIn("--max-tokens", err)


if __name__ == "__main__":
    unittest.main()
