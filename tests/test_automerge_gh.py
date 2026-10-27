"""Tests for the GitHub-facing parts of tools/automerge.py (_gh, hae_pr, hae_tiedostot, yhteenveto, main).

The ``gh`` command is never run: ``subprocess.run`` / ``_gh`` are replaced with fakes.
"""
import contextlib
import io
import json
import os
import subprocess
import tempfile
import unittest
import unittest.mock as mock

from tools import automerge as am


def done(stdout="", stderr="", returncode=0):
    return subprocess.CompletedProcess(args=["gh"], returncode=returncode, stdout=stdout, stderr=stderr)


def quiet(fn, *args, **kwargs):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        result = fn(*args, **kwargs)
    return result, out.getvalue()


class GhTests(unittest.TestCase):
    def test_runs_gh_with_args_and_timeout(self):
        with mock.patch.object(am.subprocess, "run", return_value=done("x")) as run:
            result = am._gh("api", "foo")
        run.assert_called_once_with(["gh", "api", "foo"], capture_output=True, text=True, timeout=120)
        self.assertEqual(result.stdout, "x")


class HaePrTests(unittest.TestCase):
    PAYLOAD = {
        "draft": True,
        "head": {"repo": {"full_name": "bot/repo"}, "ref": "yo/testit-x"},
        "base": {"repo": {"full_name": "org/repo"}, "ref": "main"},
        "labels": [{"name": "a"}, {"name": "ei-automerge"}],
        "user": {"login": "someone"},
    }

    def test_maps_api_fields(self):
        with mock.patch.object(am, "_gh", return_value=done(json.dumps(self.PAYLOAD))) as gh:
            pr = am.hae_pr("org/repo", "7")
        gh.assert_called_once_with("api", "repos/org/repo/pulls/7")
        self.assertEqual(pr, {
            "draft": True,
            "head_repo": "bot/repo",
            "base_repo": "org/repo",
            "labels": ["a", "ei-automerge"],
            "author": "someone",
            "head_ref": "yo/testit-x",
            "base_ref": "main",
        })

    def test_missing_fields_become_none_or_empty(self):
        # e.g. a deleted fork: head.repo is null
        payload = {"head": {"repo": None, "ref": "x"}, "base": {}, "user": None}
        with mock.patch.object(am, "_gh", return_value=done(json.dumps(payload))):
            pr = am.hae_pr("org/repo", "7")
        self.assertEqual(pr, {"draft": False, "head_repo": None, "base_repo": None, "labels": [],
                              "author": None, "head_ref": "x", "base_ref": None})

    def test_deleted_fork_is_never_accepted(self):
        payload = {"head": {"repo": None, "ref": "yo/readme-x"},
                   "base": {"repo": {"full_name": "org/repo"}, "ref": "main"},
                   "user": {"login": am.BOTTI}}
        with mock.patch.object(am, "_gh", return_value=done(json.dumps(payload))):
            pr = am.hae_pr("org/repo", "7")
        sallittu, syyt = am.arvioi(pr, [{"filename": "README.md", "status": "modified", "changes": 1,
                                         "previous_filename": None}])
        self.assertFalse(sallittu)
        self.assertTrue(syyt)

    def test_failure_raises_with_tail_of_stderr(self):
        stderr = "x" * 500 + "THE-END"
        with mock.patch.object(am, "_gh", return_value=done(stderr=stderr + "\n", returncode=1)):
            with self.assertRaises(RuntimeError) as cm:
                am.hae_pr("org/repo", "7")
        message = str(cm.exception)
        self.assertIn("THE-END", message)
        self.assertLess(len(message), 300)

    def test_invalid_json_raises_value_error(self):
        with mock.patch.object(am, "_gh", return_value=done("not json")):
            with self.assertRaises(ValueError):
                am.hae_pr("org/repo", "7")


class HaeTiedostotTests(unittest.TestCase):
    def test_parses_one_json_object_per_line_and_skips_blanks(self):
        a = {"filename": "a.md", "status": "added", "changes": 2}
        b = {"filename": "b.md", "status": "modified", "changes": 3}
        stdout = json.dumps(a) + "\n\n   \n" + json.dumps(b) + "\n"
        with mock.patch.object(am, "_gh", return_value=done(stdout)) as gh:
            result = am.hae_tiedostot("org/repo", "7")
        self.assertEqual(result, [a, b])
        args = gh.call_args.args
        self.assertEqual(args[0], "api")
        self.assertIn("--paginate", args)
        self.assertIn("repos/org/repo/pulls/7/files?per_page=100", args)

    def test_empty_output_gives_empty_list(self):
        with mock.patch.object(am, "_gh", return_value=done("")):
            self.assertEqual(am.hae_tiedostot("org/repo", "7"), [])

    def test_failure_raises(self):
        with mock.patch.object(am, "_gh", return_value=done(stderr="nope", returncode=1)):
            with self.assertRaises(RuntimeError) as cm:
                am.hae_tiedostot("org/repo", "7")
        self.assertIn("nope", str(cm.exception))

    def test_invalid_json_line_raises_value_error(self):
        with mock.patch.object(am, "_gh", return_value=done("{broken")):
            with self.assertRaises(ValueError):
                am.hae_tiedostot("org/repo", "7")


class YhteenvetoTests(unittest.TestCase):
    def test_prints_without_summary_file(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            _, out = quiet(am.yhteenveto, "hei")
        self.assertEqual(out, "hei\n")

    def test_appends_to_step_summary_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "summary.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write("alku\n")
            with mock.patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": path}, clear=True):
                _, out = quiet(am.yhteenveto, "rivi")
            with open(path, encoding="utf-8") as f:
                self.assertEqual(f.read(), "alku\nrivi\n")
        self.assertEqual(out, "rivi\n")

    def test_unwritable_summary_file_is_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            missing = os.path.join(d, "no-such-dir", "summary.md")
            with mock.patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": missing}, clear=True):
                _, out = quiet(am.yhteenveto, "viesti")
        self.assertEqual(out, "viesti\n")


class MainFlowTests(unittest.TestCase):
    ENV = {"GITHUB_REPOSITORY": "org/repo", "PR_NUMBER": "12"}
    OK_PR = {"draft": False, "head_repo": "org/repo", "base_repo": "org/repo", "labels": [],
             "author": "someone", "head_ref": "fix-1", "base_ref": "main"}
    DOC = [{"filename": "README.md", "status": "modified", "changes": 1, "previous_filename": None}]
    CODE = [{"filename": "src/app.py", "status": "modified", "changes": 1, "previous_filename": None}]

    def run_main(self, pr, tiedostot, gh_result=None, env=None):
        gh = mock.Mock(return_value=gh_result or done())
        with mock.patch.dict(os.environ, env or self.ENV, clear=True), \
                mock.patch.object(am, "hae_pr", return_value=pr), \
                mock.patch.object(am, "hae_tiedostot", return_value=tiedostot), \
                mock.patch.object(am, "_gh", gh):
            code, out = quiet(am.main)
        return code, out, gh

    def test_allowed_pr_enables_auto_merge(self):
        code, out, gh = self.run_main(self.OK_PR, self.DOC)
        self.assertEqual(code, 0)
        gh.assert_called_once_with("pr", "merge", "12", "--auto", "--squash", "--repo", "org/repo")
        self.assertIn("PR #12 kelpaa", out)

    def test_enabling_auto_merge_failure_returns_1(self):
        code, out, gh = self.run_main(self.OK_PR, self.DOC, gh_result=done(stderr="denied", returncode=1))
        self.assertEqual(code, 1)
        self.assertIn("kytkenta epaonnistui", out)
        self.assertIn("denied", out)

    def test_rejected_pr_disables_auto_merge_and_lists_reasons(self):
        code, out, gh = self.run_main(self.OK_PR, self.CODE)
        self.assertEqual(code, 0)
        gh.assert_called_once_with("pr", "merge", "12", "--disable-auto", "--repo", "org/repo")
        self.assertIn("ei kelpaa", out)
        self.assertIn("ei sallittu tyyppi: src/app.py", out)
        self.assertIn("Auto-merge poistettu", out)

    def test_rejected_pr_returns_0_even_if_disable_fails(self):
        code, _, _ = self.run_main(self.OK_PR, self.CODE, gh_result=done(stderr="x", returncode=1))
        self.assertEqual(code, 0)

    def test_value_error_from_evaluation_returns_1_and_does_nothing(self):
        gh = mock.Mock()
        with mock.patch.dict(os.environ, self.ENV, clear=True), \
                mock.patch.object(am, "hae_pr", return_value=self.OK_PR), \
                mock.patch.object(am, "hae_tiedostot", side_effect=ValueError("bad json")), \
                mock.patch.object(am, "_gh", gh):
            code, out = quiet(am.main)
        self.assertEqual(code, 1)
        gh.assert_not_called()
        self.assertIn("arviointi epaonnistui", out)
        self.assertIn("bad json", out)

    def test_missing_env_returns_2(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            code, out = quiet(am.main)
        self.assertEqual(code, 2)
        self.assertIn("virheellinen", out)

    def test_repo_and_number_are_validated_strictly(self):
        for repo, number in (("org/repo/extra", "1"), ("org", "1"), ("org/repo;rm", "1"),
                             ("org/repo", ""), ("org/repo", "1 2"), ("org/repo", "-1")):
            with self.subTest(repo=repo, number=number), \
                    mock.patch.dict(os.environ, {"GITHUB_REPOSITORY": repo, "PR_NUMBER": number}, clear=True), \
                    mock.patch.object(am, "hae_pr") as hae:
                code, _ = quiet(am.main)
                self.assertEqual(code, 2)
                hae.assert_not_called()


class ArvioiGapTests(unittest.TestCase):
    """Remaining edge cases of arvioi that the other test files do not reach."""

    def bot_pr(self, head_ref="yo/testit-x"):
        return {"draft": False, "head_repo": "agenttifirma-bot/repo", "base_repo": "org/repo", "labels": [],
                "author": "agenttifirma-bot", "head_ref": head_ref, "base_ref": "main"}

    def f(self, name, status="modified", changes=1, previous=None):
        return {"filename": name, "status": status, "changes": changes, "previous_filename": previous}

    def test_missing_changes_counts_as_zero(self):
        sallittu, syyt = am.arvioi(self.bot_pr(), [{"filename": "a.py", "status": "added"}])
        self.assertTrue(sallittu, syyt)

    def test_bot_fork_previous_filename_must_be_safe_too(self):
        sallittu, syyt = am.arvioi(self.bot_pr(), [self.f("src/new.py", previous="yojono/old.py")])
        self.assertFalse(sallittu)
        self.assertTrue(any("yojono/old.py" in s for s in syyt))

    def test_bot_fork_removed_and_renamed_files_rejected(self):
        for status in ("removed", "renamed"):
            with self.subTest(status=status):
                self.assertFalse(am.arvioi(self.bot_pr(), [self.f("src/a.py", status=status)])[0])

    def test_bot_readme_branch_md_in_subdir_is_ok(self):
        sallittu, syyt = am.arvioi(self.bot_pr("yo/readme-x"), [self.f("docs/a.md")])
        self.assertTrue(sallittu, syyt)

    def test_bot_readme_branch_reports_each_non_md_file(self):
        sallittu, syyt = am.arvioi(self.bot_pr("yo/readme-x"), [self.f("a.py"), self.f("b.js")])
        self.assertFalse(sallittu)
        self.assertTrue(any("a.py" in s for s in syyt))
        self.assertTrue(any("b.js" in s for s in syyt))

    def test_bot_code_branch_allows_js_and_ts(self):
        files = [self.f("web/a.js"), self.f("web/b.ts"), self.f("web/c.tsx"), self.f("web/d.mjs")]
        sallittu, syyt = am.arvioi(self.bot_pr("yo/ominaisuus-x"), files)
        self.assertTrue(sallittu, syyt)

    def test_bot_code_branch_rejects_unlisted_suffix_and_unsafe_chars(self):
        self.assertFalse(am.arvioi(self.bot_pr(), [self.f("src/a.sh")])[0])
        self.assertFalse(am.arvioi(self.bot_pr(), [self.f("src/a b.py")])[0])

    def test_bot_code_line_limit_counts_sum_over_files(self):
        files = [self.f("a.py", changes=200), self.f("b.py", changes=201)]
        self.assertFalse(am.arvioi(self.bot_pr(), files)[0])

    def test_too_many_files_boundary(self):
        at_limit = [self.f("f%d.md" % i) for i in range(20)]
        pr = {"draft": False, "head_repo": "org/repo", "base_repo": "org/repo", "labels": [],
              "author": "someone", "head_ref": "fix", "base_ref": "main"}
        self.assertTrue(am.arvioi(pr, at_limit)[0])

    def test_reasons_are_not_duplicated_for_same_protected_path(self):
        pr = {"draft": False, "head_repo": "org/repo", "base_repo": "org/repo", "labels": [],
              "author": "someone", "head_ref": "fix", "base_ref": "main"}
        _, syyt = am.arvioi(pr, [self.f(".env.md", previous=".env.md")])
        self.assertEqual(len(syyt), len(set(syyt)))


if __name__ == "__main__":
    unittest.main()
