"""Edge cases the other test files leave open: core branches, error mapping, CLI failure paths,
automerge rules and version consistency."""
import contextlib
import io
import json
import os
import re
import tempfile
import unittest
import unittest.mock as mock

import llm_stopcheck as sc
from llm_stopcheck import cli
from tools import automerge as am

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resp(content="hello", finish="stop", native=None, usage=None, **message):
    msg = {"role": "assistant", "content": content}
    msg.update(message)
    choice = {"index": 0, "finish_reason": finish, "message": msg}
    if native is not None:
        choice["native_finish_reason"] = native
    out = {"choices": [choice]}
    if usage is not None:
        out["usage"] = usage
    return out


class CoreBranchTests(unittest.TestCase):
    def test_native_filter_reason_overrides_stop(self):
        v = sc.check(resp("", finish="stop", native="SAFETY"))
        self.assertEqual((v.code, v.hint, v.native_finish_reason), (sc.CONTENT_FILTERED, sc.HINT_FAIL, "safety"))

    def test_truncation_wins_over_native_filter(self):
        v = sc.check(resp("cut", finish="length", native="safety"))
        self.assertEqual(v.code, sc.TRUNCATED)
        self.assertNotIn("native_finish_reason", v.reason)

    def test_no_mismatch_note_when_finish_itself_is_truncated(self):
        v = sc.check(resp("cut", finish="length", native="stop"))
        self.assertNotIn("native_finish_reason", v.reason)

    def test_refusal_finish_reason_without_refusal_field(self):
        v = sc.check(resp(None, finish="refusal"))
        self.assertEqual((v.code, v.hint), (sc.REFUSAL, sc.HINT_FAIL))

    def test_blank_refusal_text_is_not_a_refusal(self):
        self.assertTrue(sc.check(resp("hi", refusal="   ")).ok)

    def test_tool_call_truncated_via_legacy_function_call(self):
        r = resp(None, finish="length", function_call={"name": "f", "arguments": "{"})
        self.assertEqual(sc.check(r).code, sc.TOOL_CALL_TRUNCATED)

    def test_zero_reasoning_tokens_and_blank_reasoning_text_is_truncated_empty(self):
        usage = {"completion_tokens_details": {"reasoning_tokens": 0}}
        self.assertEqual(sc.check(resp("", finish="length", usage=usage)).code, sc.TRUNCATED_EMPTY)
        r = resp("", finish="length", reasoning="   ")
        self.assertEqual(sc.check(r).code, sc.TRUNCATED_EMPTY)

    def test_non_dict_details_fall_back_to_flat_reasoning_tokens(self):
        usage = {"completion_tokens_details": "x", "reasoning_tokens": 7}
        self.assertEqual(sc.check(resp(usage=usage)).reasoning_tokens, 7)

    def test_non_dict_usage_is_ignored(self):
        v = sc.check(resp(usage="x"), max_tokens=10)
        self.assertTrue(v.ok)
        self.assertIsNone(v.completion_tokens)

    def test_message_that_is_not_an_object(self):
        r = {"choices": [{"finish_reason": "stop", "message": "x"}]}
        self.assertEqual(sc.check(r).code, sc.EMPTY_OUTPUT)
        self.assertTrue(sc.check(r, require_text=False).ok)

    def test_choices_must_be_a_list(self):
        v = sc.check({"choices": {"0": {}}}, max_tokens=5)
        self.assertEqual((v.code, v.hint, v.max_tokens), (sc.MALFORMED_RESPONSE, sc.HINT_RETRY, 5))

    def test_empty_content_list_is_empty_output(self):
        self.assertEqual(sc.check(resp([])).code, sc.EMPTY_OUTPUT)

    def test_limit_warning_boundaries(self):
        near = sc.check(resp(usage={"completion_tokens": 50}), max_tokens=100, near_limit=0.5)
        self.assertEqual(near.warnings, ["near_limit"])
        over = sc.check(resp(usage={"completion_tokens": 150}), max_tokens=100)
        self.assertEqual(over.warnings, ["at_limit"])
        no_cap = sc.check(resp(usage={"completion_tokens": 150}), max_tokens=0)
        self.assertEqual(no_cap.warnings, [])

    def test_failed_verdicts_carry_no_limit_warnings(self):
        v = sc.check(resp("cut", finish="length", usage={"completion_tokens": 100}), max_tokens=100)
        self.assertEqual(v.warnings, [])

    def test_verdict_reports_normalised_finish_and_content_chars(self):
        v = sc.check(resp("hello", finish=" STOP "))
        self.assertEqual((v.finish_reason, v.content_chars), ("stop", 5))

    def test_extract_text_uses_first_choice_only(self):
        r = resp("first")
        r["choices"].append({"message": {"content": "second"}})
        self.assertEqual(sc.extract_text(r), "first")
        self.assertEqual(sc.extract_text(resp(None)), "")

    def test_converter_falls_through_to_next_method(self):
        class Obj:
            def model_dump(self):
                return ["not a dict"]

            def dict(self):
                return resp("via dict")
        self.assertEqual(sc.extract_text(Obj()), "via dict")


class ErrorMappingTests(unittest.TestCase):
    EXPECTED = {
        sc.TRUNCATED: sc.TruncatedError,
        sc.TRUNCATED_EMPTY: sc.TruncatedError,
        sc.REASONING_EXHAUSTED: sc.TruncatedError,
        sc.TOOL_CALL_TRUNCATED: sc.TruncatedError,
        sc.EMPTY_OUTPUT: sc.EmptyOutputError,
        sc.CONTENT_FILTERED: sc.ContentFilteredError,
        sc.REFUSAL: sc.RefusalError,
        sc.MISSING_FINISH_REASON: sc.ProtocolError,
        sc.UNKNOWN_FINISH_REASON: sc.ProtocolError,
        sc.MALFORMED_RESPONSE: sc.ProtocolError,
    }

    def test_every_failing_code_maps_to_its_exception(self):
        self.assertEqual(set(self.EXPECTED), set(sc.ALL_CODES) - {sc.OK})
        for code, exc_type in self.EXPECTED.items():
            with self.subTest(code=code):
                with self.assertRaises(exc_type) as cm:
                    sc.raise_for_verdict(sc.Verdict(False, code, "why", hint=sc.HINT_RETRY))
                self.assertIs(type(cm.exception), exc_type)
                self.assertIsInstance(cm.exception, sc.StopcheckError)
                self.assertEqual((cm.exception.code, cm.exception.hint), (code, sc.HINT_RETRY))
                self.assertEqual(str(cm.exception), "%s: why" % code)

    def test_unknown_code_falls_back_to_base_error(self):
        with self.assertRaises(sc.StopcheckError) as cm:
            sc.raise_for_verdict(sc.Verdict(False, "brand_new", "x"))
        self.assertIs(type(cm.exception), sc.StopcheckError)

    def test_ok_verdict_is_returned_unchanged(self):
        v = sc.Verdict(True, sc.OK, "ok")
        self.assertIs(sc.raise_for_verdict(v), v)

    def test_ensure_forwards_kwargs_to_check(self):
        self.assertEqual(sc.ensure(resp(""), require_text=False), "")
        with self.assertRaises(sc.EmptyOutputError):
            sc.ensure(resp(""))
        self.assertEqual(sc.ensure(resp("hi", usage={"completion_tokens": 10}), max_tokens=10), "hi")

    def test_ensure_with_tool_call_returns_empty_text(self):
        self.assertEqual(sc.ensure(resp(None, finish="tool_calls", tool_calls=[{"id": "1"}])), "")


class FormatRecordTests(unittest.TestCase):
    def test_exact_line_for_truncated_response(self):
        v = sc.check(resp("x", finish="length"), max_tokens=3000)
        self.assertEqual(sc.format_record(v), "stopcheck=truncated finish=length native=- tokens=-/3000 "
                                              "reasoning_tokens=- hint=raise_max_tokens")

    def test_zero_is_not_replaced_by_dash_but_empty_string_is(self):
        v = sc.Verdict(True, sc.OK, "ok", completion_tokens=0, reasoning_tokens=0, max_tokens=5)
        line = sc.format_record(v, empty="", zero=0)
        self.assertIn("tokens=0/5", line)
        self.assertIn("reasoning_tokens=0", line)
        self.assertIn("empty=-", line)
        self.assertIn("zero=0", line)

    def test_values_are_cut_to_60_chars_and_keys_to_30(self):
        line = sc.format_record(sc.Verdict(True, sc.OK, "ok"), **{"k" * 40: "v" * 100})
        self.assertIn(" %s=%s" % ("k" * 30, "v" * 60), line)
        self.assertTrue(line.endswith("v" * 60))

    def test_allowed_punctuation_survives_and_non_ascii_is_replaced(self):
        line = sc.format_record(sc.Verdict(True, sc.OK, "ok"), model="a/b:c+d.e-f", city="café x")
        self.assertIn("model=a/b:c+d.e-f", line)
        self.assertIn("city=caf__x", line)

    def test_warnings_are_joined_and_extras_come_last(self):
        v = sc.Verdict(True, sc.OK, "ok", warnings=["a", "b"])
        line = sc.format_record(v, run="7")
        self.assertIn("warn=a_b", line)
        self.assertTrue(line.endswith("warn=a_b run=7"))


def run_cli(argv, stdin=None):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        if stdin is None:
            code = cli.main(argv)
        else:
            with mock.patch("sys.stdin", io.StringIO(stdin)):
                code = cli.main(argv)
    return code, out.getvalue(), err.getvalue()


class CliFailurePathTests(unittest.TestCase):
    OK_BODY = {"choices": [{"finish_reason": "stop", "message": {"content": "hi"}}]}

    def write(self, data, mode="w"):
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, mode, **({} if "b" in mode else {"encoding": "utf-8"})) as f:
            f.write(data if isinstance(data, (str, bytes)) else json.dumps(data))
        self.addCleanup(os.remove, path)
        return path

    def test_bad_option_values_exit_2(self):
        path = self.write(self.OK_BODY)
        for argv in ([path, "--max-tokens", "abc"], [path, "--near-limit", "abc"], [path, "--nope"]):
            with self.subTest(argv=argv):
                self.assertEqual(run_cli(argv)[0], 2)

    def test_invalid_json_on_stdin_exit_2(self):
        code, out, err = run_cli(["-"], stdin="{broken")
        self.assertEqual((code, out), (2, ""))
        self.assertIn("cannot read input -", err)

    def test_empty_file_non_utf8_and_directory_exit_2(self):
        self.assertEqual(run_cli([self.write("")])[0], 2)
        self.assertEqual(run_cli([self.write(b"\xff\xfe\x00", mode="wb")])[0], 2)
        self.assertEqual(run_cli([tempfile.gettempdir()])[0], 2)

    def test_non_object_json_is_a_failed_verdict_not_a_usage_error(self):
        code, out, _ = run_cli([self.write([1, 2])])
        self.assertEqual(code, 1)
        self.assertIn("stopcheck=malformed_response", out)

    def test_json_output_has_sorted_keys_and_warnings_still_go_to_stderr(self):
        body = dict(self.OK_BODY, usage={"completion_tokens": 10})
        code, out, err = run_cli([self.write(body), "--json", "--max-tokens", "10"])
        data = json.loads(out)
        self.assertEqual(code, 0)
        self.assertEqual(list(data), sorted(data))
        self.assertEqual(data["warnings"], ["at_limit"])
        self.assertIn("warnings: at_limit", err)

    def test_json_mode_prints_one_object_per_file(self):
        good, bad = self.write(self.OK_BODY), self.write({"choices": [{"finish_reason": "length", "message": {}}]})
        code, out, _ = run_cli([good, bad, "--json"])
        self.assertEqual(code, 1)
        self.assertEqual([json.loads(line)["code"] for line in out.splitlines()], ["ok", "truncated_empty"])

    def test_near_limit_of_exactly_one_is_accepted(self):
        path = self.write(dict(self.OK_BODY, usage={"completion_tokens": 99}))
        code, out, err = run_cli([path, "--max-tokens", "100", "--near-limit", "1"])
        self.assertEqual((code, err), (0, ""))
        self.assertIn("tokens=99/100", out)


class AutomergePathRuleTests(unittest.TestCase):
    def test_polku_ok_only_accepts_singular_test_dir_and_top_level_or_docs_markdown(self):
        for path in ("tests/test_x.py", "src/notes.md", "docs/x.txt", "docs/a b.md", "docs/../x.md", "Token.md",
                     "monkey.md"):
            with self.subTest(path=path):
                self.assertFalse(am.polku_ok(path)[0])

    def test_polku_ok_reasons(self):
        self.assertEqual(am.polku_ok("tools/x.py"), (False, "suojattu polku: tools/x.py"))
        self.assertEqual(am.polku_ok("src/app.py"), (False, "ei sallittu tyyppi: src/app.py"))

    def test_botin_koodi_ok_reasons_and_missing_path(self):
        self.assertEqual(am.botin_koodi_ok(None), (False, "ei sallittu polku: None"))
        self.assertFalse(am.botin_koodi_ok("")[0])
        self.assertEqual(am.botin_koodi_ok("src/a.txt"), (False, "ei sallittu paate: src/a.txt"))
        self.assertEqual(am.botin_koodi_ok("tools/a.py"), (False, "suojattu polku: tools/a.py"))

    def test_botin_koodi_ok_suffix_list(self):
        for suffix in am.KOODIPAATTEET:
            with self.subTest(suffix=suffix):
                self.assertTrue(am.botin_koodi_ok("src/a" + suffix)[0])
        self.assertFalse(am.botin_koodi_ok("src/a.PY")[0])

    def test_botin_koodi_ok_rejects_protected_names_in_subdirectories(self):
        for path in ("pkg/pyproject.toml", "pkg/setup.py", "web/Dockerfile", "web/package.json", "a/poetry.lock",
                     "agentti/portti.py", "agentti/ehdotukset.py", "agentti/maaritys_x.py", "src/key_store.py",
                     "src/api_token.py", "a/../b.py", "src/.env.py"):
            with self.subTest(path=path):
                self.assertFalse(am.botin_koodi_ok(path)[0])


class AutomergeBotBranchTests(unittest.TestCase):
    def bot(self, **kw):
        base = {"draft": False, "head_repo": "agenttifirma-bot/repo", "base_repo": "org/repo", "labels": [],
                "author": "agenttifirma-bot", "head_ref": "yo/testit-x", "base_ref": "main"}
        base.update(kw)
        return base

    def f(self, name, status="modified", changes=1):
        return {"filename": name, "status": status, "changes": changes, "previous_filename": None}

    def test_all_reasons_accumulate_for_a_wrong_fork(self):
        p = self.bot(author="human", base_ref="dev", head_ref="feature")
        self.assertEqual(am.botin_forkki_syyt(p), [
            "PR ei ole saman repon haarasta", "base ei ole main eika master",
            "botin haaran nimi ei ole yo/(readme|testit|ominaisuus)-*"])

    def test_empty_metadata_is_rejected_with_three_reasons(self):
        self.assertEqual(len(am.botin_forkki_syyt({})), 3)

    def test_missing_head_repo_is_not_a_bot_fork(self):
        self.assertIn("forkki ei ole botin oma fork samasta reposta", am.botin_forkki_syyt(self.bot(head_repo=None)))

    def test_repo_name_comparison_ignores_case(self):
        self.assertEqual(am.botin_forkki_syyt(self.bot(head_repo="AgenttiFirma-Bot/REPO")), [])

    def test_branch_name_rules(self):
        accepted = ("yo/readme-a", "yo/testit-1.2_x-y", "yo/ominaisuus-" + "a" * 60)
        rejected = ("yo/testit-", "yo/readme-UPPER", "yo/readme-.x", "yo/readme-a/b", "yo/readme-" + "a" * 61,
                    "readme-a", "yo/other-a")
        for ref in accepted:
            with self.subTest(ref=ref):
                self.assertEqual(am.botin_forkki_syyt(self.bot(head_ref=ref)), [])
        for ref in rejected:
            with self.subTest(ref=ref):
                self.assertTrue(am.botin_forkki_syyt(self.bot(head_ref=ref)))

    def test_bot_readme_branch_only_allows_top_level_or_docs_markdown(self):
        readme = self.bot(head_ref="yo/readme-x")
        self.assertFalse(am.arvioi(readme, [self.f("src/notes.md")])[0])
        self.assertFalse(am.arvioi(readme, [self.f("tools/x.md")])[0])
        self.assertFalse(am.arvioi(readme, [self.f(".github/README.md")])[0])
        self.assertTrue(am.arvioi(readme, [self.f("README.md"), self.f("docs/a.md")])[0])

    def test_bot_code_branch_allows_markdown_anywhere_unprotected(self):
        sallittu, syyt = am.arvioi(self.bot(), [self.f("src/notes.md")])
        self.assertTrue(sallittu, syyt)

    def test_bot_fork_draft_and_opt_out_are_rejected(self):
        self.assertIn("luonnos", am.arvioi(self.bot(draft=True), [self.f("a.py")])[1])
        self.assertIn("tunniste ei-automerge", am.arvioi(self.bot(labels=["ei-automerge"]), [self.f("a.py")])[1])

    def test_bot_fork_file_count_boundary_and_reason_text(self):
        self.assertTrue(am.arvioi(self.bot(), [self.f("f%d.py" % i) for i in range(20)])[0])
        sallittu, syyt = am.arvioi(self.bot(), [self.f("f%d.py" % i) for i in range(21)])
        self.assertFalse(sallittu)
        self.assertIn("liian monta tiedostoa (21)", syyt)

    def test_bot_fork_line_limit_reason_text(self):
        self.assertIn("liian suuri muutos (401 rivia)", am.arvioi(self.bot(), [self.f("a.py", changes=401)])[1])

    def test_other_file_statuses_are_rejected_with_reason(self):
        for status in ("copied", "changed", "unchanged"):
            with self.subTest(status=status):
                self.assertIn("tiedoston tila %s: x.md" % status,
                              am.arvioi(self.bot(head_ref="yo/readme-x"), [self.f("x.md", status=status)])[1])


class AutomergeSameRepoTests(unittest.TestCase):
    PR = {"draft": False, "head_repo": "org/repo", "base_repo": "org/repo", "labels": [],
          "author": "someone", "head_ref": "fix", "base_ref": "main"}

    def f(self, name, status="modified", changes=1):
        return {"filename": name, "status": status, "changes": changes, "previous_filename": None}

    def test_labels_none_is_treated_as_no_labels(self):
        self.assertTrue(am.arvioi(dict(self.PR, labels=None), [self.f("README.md")])[0])

    def test_added_files_are_allowed(self):
        self.assertTrue(am.arvioi(self.PR, [self.f("docs/new.md", status="added")])[0])

    def test_no_files_reason(self):
        self.assertEqual(am.arvioi(self.PR, []), (False, ["ei tiedostoja"]))

    def test_reasons_aggregate_over_independent_problems(self):
        pr = dict(self.PR, draft=True, labels=["ei-automerge"])
        sallittu, syyt = am.arvioi(pr, [self.f("src/a.py", status="removed")])
        self.assertFalse(sallittu)
        self.assertEqual(syyt, ["luonnos", "tunniste ei-automerge", "tiedoston tila removed: src/a.py",
                                "ei sallittu tyyppi: src/a.py"])

    def test_line_limit_reason_text_and_readme_branch_name_grants_nothing(self):
        pr = dict(self.PR, head_ref="yo/readme-x")
        self.assertIn("liian suuri muutos (201 rivia)", am.arvioi(pr, [self.f("README.md", changes=201)])[1])
        self.assertFalse(am.arvioi(pr, [self.f("src/notes.md")])[0])


class AutomergeMainExtraTests(unittest.TestCase):
    ENV = {"GITHUB_REPOSITORY": "org/repo", "PR_NUMBER": "3"}
    BOT_PR = {"draft": False, "head_repo": "agenttifirma-bot/repo", "base_repo": "org/repo", "labels": [],
              "author": "agenttifirma-bot", "head_ref": "yo/testit-x", "base_ref": "main"}
    TEST_FILE = [{"filename": "tests/test_x.py", "status": "added", "changes": 5, "previous_filename": None}]

    def run_main(self, env, **patches):
        with mock.patch.dict(os.environ, env, clear=True), contextlib.redirect_stdout(io.StringIO()) as out:
            for name, value in patches.items():
                self.enterContext(mock.patch.object(am, name, **value))
            return am.main(), out.getvalue()

    def test_bot_testit_fork_gets_auto_merge(self):
        gh = mock.Mock(return_value=mock.Mock(returncode=0, stderr=""))
        code, out = self.run_main(self.ENV, hae_pr={"return_value": self.BOT_PR},
                                  hae_tiedostot={"return_value": self.TEST_FILE}, _gh={"new": gh})
        self.assertEqual(code, 0)
        gh.assert_called_once_with("pr", "merge", "3", "--auto", "--squash", "--repo", "org/repo")
        self.assertIn("PR #3 kelpaa", out)

    def test_file_list_failure_returns_1_and_runs_no_gh_command(self):
        gh = mock.Mock()
        code, out = self.run_main(self.ENV, hae_pr={"return_value": self.BOT_PR},
                                  hae_tiedostot={"side_effect": RuntimeError("files down")}, _gh={"new": gh})
        self.assertEqual(code, 1)
        gh.assert_not_called()
        self.assertIn("files down", out)

    def test_decision_is_appended_to_step_summary(self):
        gh = mock.Mock(return_value=mock.Mock(returncode=0, stderr=""))
        with tempfile.TemporaryDirectory() as d:
            summary = os.path.join(d, "s.md")
            env = dict(self.ENV, GITHUB_STEP_SUMMARY=summary)
            self.run_main(env, hae_pr={"return_value": self.BOT_PR},
                          hae_tiedostot={"return_value": [{"filename": "a.sh", "status": "added", "changes": 1,
                                                           "previous_filename": None}]}, _gh={"new": gh})
            with open(summary, encoding="utf-8") as fh:
                text = fh.read()
        self.assertIn("ei kelpaa automaattiseen yhdistamiseen", text)
        self.assertIn("ei sallittu paate: a.sh", text)

    def test_files_are_fetched_with_jq_flag(self):
        gh = mock.Mock(return_value=mock.Mock(returncode=0, stdout="", stderr=""))
        with mock.patch.object(am, "_gh", gh):
            am.hae_tiedostot("org/repo", "3")
        args = gh.call_args.args
        self.assertEqual(args[args.index("--jq") + 1], ".[]")


class VersionConsistencyTests(unittest.TestCase):
    def test_pyproject_version_matches_package_version(self):
        with open(os.path.join(ROOT, "pyproject.toml"), encoding="utf-8") as fh:
            match = re.search(r'^version\s*=\s*"([^"]+)"', fh.read(), re.M)
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), sc.__version__)

    def test_changelog_documents_current_version_or_unreleased(self):
        with open(os.path.join(ROOT, "CHANGELOG.md"), encoding="utf-8") as fh:
            headings = re.findall(r"^## (.+)$", fh.read(), re.M)
        self.assertIn(sc.__version__, headings)
        self.assertEqual(headings[0], "Unreleased")


if __name__ == "__main__":
    unittest.main()
