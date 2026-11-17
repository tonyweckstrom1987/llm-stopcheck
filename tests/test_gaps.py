import contextlib
import io
import json
import subprocess
import unittest
from unittest import mock

import llm_stopcheck as sc
from llm_stopcheck import cli, core, errors
from tools import automerge as am

SHA = "a" * 40


def resp(content="hello", finish="stop", **message):
    msg = {"role": "assistant", "content": content}
    msg.update(message)
    return {"choices": [{"finish_reason": finish, "message": msg}]}


class CoreGapTests(unittest.TestCase):
    def test_verdict_defaults_do_not_share_warnings(self):
        a, b = core.Verdict(True, "ok", "ok"), core.Verdict(True, "ok", "ok")
        a.warnings.append("x")
        self.assertEqual(b.warnings, [])
        self.assertEqual((a.content_chars, a.hint), (0, None))

    def test_blank_finish_reason_counts_as_missing(self):
        for blank in ("", "   ", None):
            self.assertEqual(sc.check(resp(finish=blank)).code, sc.MISSING_FINISH_REASON, repr(blank))

    def test_refusal_via_finish_reason_and_whitespace_refusal_text(self):
        self.assertEqual(sc.check(resp("", finish="refusal")).code, sc.REFUSAL)
        self.assertTrue(sc.check(resp("fine", refusal="   ")).ok)

    def test_native_finish_reason_reveals_content_filter(self):
        v = sc.check(resp("x", finish="stop") | {"choices": [dict(resp()["choices"][0], native_finish_reason="SAFETY")]})
        self.assertEqual((v.code, v.hint), (sc.CONTENT_FILTERED, sc.HINT_FAIL))

    def test_truncation_wins_over_filter_and_text_over_tool_calls_is_still_tool_truncated(self):
        r = resp("partial", finish="length", tool_calls=[{"id": "1"}])
        self.assertEqual(sc.check(r).code, sc.TOOL_CALL_TRUNCATED)

    def test_require_text_false_does_not_excuse_truncated_empty(self):
        self.assertEqual(sc.check(resp("", finish="length"), require_text=False).code, sc.TRUNCATED_EMPTY)

    def test_message_that_is_not_an_object_gives_empty_output(self):
        r = {"choices": [{"finish_reason": "stop", "message": "oops"}]}
        v = sc.check(r)
        self.assertEqual((v.code, v.content_chars), (sc.EMPTY_OUTPUT, 0))

    def test_limit_warning_boundaries(self):
        def warn(tokens, cap, **kw):
            r = resp() | {"usage": {"completion_tokens": tokens}}
            return sc.check(r, max_tokens=cap, **kw).warnings
        self.assertEqual(warn(95, 100), ["near_limit"])
        self.assertEqual(warn(94, 100), [])
        self.assertEqual(warn(101, 100), ["at_limit"])
        self.assertEqual(warn(50, 100, near_limit=0.5), ["near_limit"])
        self.assertEqual(warn(10, 0), [])

    def test_verdict_carries_content_chars_and_max_tokens_on_malformed(self):
        self.assertEqual(sc.check(resp("abcd")).content_chars, 4)
        v = sc.check(None, max_tokens=7)
        self.assertEqual((v.max_tokens, v.hint, v.ok), (7, sc.HINT_RETRY, False))


class ErrorsGapTests(unittest.TestCase):
    def test_every_failing_code_maps_to_its_exception_class(self):
        expected = {
            sc.TRUNCATED: sc.TruncatedError, sc.TRUNCATED_EMPTY: sc.TruncatedError,
            sc.REASONING_EXHAUSTED: sc.TruncatedError, sc.TOOL_CALL_TRUNCATED: sc.TruncatedError,
            sc.EMPTY_OUTPUT: sc.EmptyOutputError, sc.CONTENT_FILTERED: sc.ContentFilteredError,
            sc.REFUSAL: sc.RefusalError, sc.MISSING_FINISH_REASON: sc.ProtocolError,
            sc.UNKNOWN_FINISH_REASON: sc.ProtocolError, sc.MALFORMED_RESPONSE: sc.ProtocolError,
        }
        self.assertEqual(set(expected), set(sc.ALL_CODES) - {sc.OK})
        for code, cls in expected.items():
            with self.assertRaises(cls) as cm:
                sc.raise_for_verdict(core.Verdict(False, code, "r", hint="h"))
            self.assertEqual((cm.exception.code, cm.exception.hint), (code, "h"))
            self.assertIsInstance(cm.exception, sc.StopcheckError)

    def test_ok_verdict_is_returned_unchanged(self):
        v = core.Verdict(True, sc.OK, "ok")
        self.assertIs(sc.raise_for_verdict(v), v)

    def test_unknown_code_falls_back_to_base_class(self):
        with self.assertRaises(sc.StopcheckError) as cm:
            sc.raise_for_verdict(core.Verdict(False, "brand_new", "why"))
        self.assertIs(type(cm.exception), sc.StopcheckError)
        self.assertEqual(str(cm.exception), "brand_new: why")

    def test_ensure_passes_kwargs_to_check(self):
        self.assertEqual(sc.ensure(resp(""), require_text=False), "")
        with self.assertRaises(sc.EmptyOutputError):
            sc.ensure(resp(""))

    def test_format_record_truncates_long_values_and_sanitises_extra_keys(self):
        v = core.Verdict(False, "truncated", "r", finish_reason="x" * 100, completion_tokens=0)
        line = errors.format_record(v, **{"bad key\n": "v", "n": 0, "e": ""})
        self.assertIn("finish=" + "x" * 60 + " ", line)
        self.assertIn("tokens=0/-", line)
        self.assertIn("bad_key_=v", line)
        self.assertIn("n=0", line)
        self.assertIn("e=-", line)
        self.assertNotIn("warn=", line)


class CliGapTests(unittest.TestCase):
    def run_cli(self, argv, stdin=None):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), \
                mock.patch("sys.stdin", io.StringIO(stdin or "")):
            code = cli.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_non_numeric_max_tokens_is_a_usage_error(self):
        code, _, err = self.run_cli(["-", "--max-tokens", "abc"])
        self.assertEqual(code, 2)
        self.assertIn("max-tokens", err)

    def test_invalid_json_on_stdin_exit_2(self):
        code, out, err = self.run_cli(["-"], stdin="{not json")
        self.assertEqual((code, out), (2, ""))
        self.assertIn("cannot read input -", err)

    def test_json_output_is_sorted_single_line_and_complete(self):
        code, out, _ = self.run_cli(["-", "--json"], stdin=json.dumps(resp("hi")))
        self.assertEqual(code, 0)
        self.assertEqual(len(out.splitlines()), 1)
        data = json.loads(out)
        self.assertEqual(list(data), sorted(data))
        self.assertEqual(set(data), set(core.Verdict(True, "ok", "ok").as_dict()))

    def test_near_limit_exactly_one_is_accepted(self):
        code, _, _ = self.run_cli(["-", "--near-limit", "1"], stdin=json.dumps(resp("hi")))
        self.assertEqual(code, 0)

    def test_malformed_response_file_exit_1(self):
        code, out, _ = self.run_cli(["-"], stdin="[]")
        self.assertEqual(code, 1)
        self.assertIn("stopcheck=malformed_response", out)


def tied(nimi, changes=5, patch="@@ -1 +1 @@\n-a\n+b", **lisa):
    return dict({"filename": nimi, "status": "modified", "changes": changes, "patch": patch}, **lisa)


PR = {"draft": False, "head_repo": "o/r", "base_repo": "o/r", "labels": [], "author": "tonyweckstrom1987",
      "head_ref": "korjaus/x", "base_ref": "main", "head_sha": SHA}
OK = [{"user": "agenttifirma-bot",
       "body": "<!-- riippumaton-tarkastus v1 sha=%s paatos=HYVAKSYTTY -->" % SHA}]


class AutomergeGapTests(unittest.TestCase):
    def test_tarkastajan_paatos_ilman_kommentteja(self):
        for tyhja in (None, [], ()):
            self.assertIsNone(am.tarkastajan_paatos(tyhja, SHA))
        self.assertIsNone(am.tarkastajan_paatos([{"body": OK[0]["body"]}], SHA))

    def test_lisatyt_rivit_ohittaa_otsikon_ja_poistot(self):
        patch = "--- a/x\n+++ b/x\n@@ -1,2 +1,2 @@\n-vanha\n+uusi\n kontekstia\n++kaksi"
        self.assertEqual(am._lisatyt_rivit(patch), ["uusi", "+kaksi"])
        self.assertEqual(am._lisatyt_rivit(None), [])

    def test_rajattu_tyhja_tai_puuttuva_polku(self):
        for polku in ("", None):
            self.assertEqual(am.rajattu("r", polku), "polku")

    def test_botin_forkki_syyt(self):
        bot = dict(PR, author="AgenttiFirma-Bot", head_repo="AGENTTIFIRMA-BOT/r", head_ref="yo/ok-1")
        self.assertEqual(am.botin_forkki_syyt(bot), [])
        self.assertEqual(am.botin_forkki_syyt(dict(bot, base_repo="ilman-omistajaa")),
                         ["forkki ei ole botin oma fork samasta reposta"])
        self.assertEqual(am.botin_forkki_syyt(dict(bot, author="joku")), ["forkista vain botin PR"])

    def test_botin_haaran_pituusraja(self):
        self.assertTrue(am.BOTIN_HAARA.match("yo/" + "a" * 60))
        self.assertFalse(am.BOTIN_HAARA.match("yo/" + "a" * 61))
        self.assertFalse(am.BOTIN_HAARA.match("yo/-a"))
        self.assertFalse(am.BOTIN_HAARA.match("yo/a\n"))

    def test_ulkopuolinen_forkki_antaa_molemmat_syyt(self):
        ok, syyt = am.arvioi(dict(PR, head_repo="x/r"), [tied("a.py")], OK)
        self.assertFalse(ok)
        self.assertEqual(syyt, ["forkista vain botin PR", "botin haaran nimi ei ole yo/<nimi>"])

    def test_sama_rajattu_nimi_raportoidaan_kerran(self):
        _, syyt = am.arvioi(PR, [tied(".github/x.yml", status="renamed", previous_filename=".github/x.yml")], OK)
        self.assertEqual(syyt, ["rajattu (tyonkulut (.github)): .github/x.yml"])

    def test_useita_syita_kertyy(self):
        ok, syyt = am.arvioi(dict(PR, draft=True, labels=["ei-automerge"], base_ref="dev"), [tied("a.py")], [])
        self.assertFalse(ok)
        self.assertEqual(syyt, ["luonnos", "base ei ole main eika master", "tunniste ei-automerge",
                                "riippumaton tarkastus: odottaa"])

    def test_tailnet_ja_cgnat_rajat(self):
        def estaa(rivi):
            return not am.arvioi(PR, [tied("a.py", patch="@@\n+" + rivi)], OK)[0]
        self.assertTrue(estaa("a 100." + "127.255.255 b"))
        self.assertTrue(estaa("172." + "31.0.1"))
        self.assertFalse(estaa("100." + "128.0.1"))
        self.assertTrue(estaa("HOST.TS" + ".NET"))

    def test_hae_funktiot_jasentavat_gh_tulosteen(self):
        pr_json = json.dumps({"draft": None, "state": "open", "head": None, "base": {"repo": {"full_name": "o/r"}, "ref": "main"},
                              "labels": [{"name": "a"}], "user": {"login": "u"}})
        rivit = "\n".join(json.dumps(x) for x in ({"user": "u", "body": "b"}, {"user": "v", "body": "c"})) + "\n\n"

        def f(args, **kw):
            osat = args[2:]
            return subprocess.CompletedProcess(args, 0, rivit if "--paginate" in osat else pr_json, "")
        with mock.patch.object(am.subprocess, "run", f):
            pr = am.hae_pr("o/r", "1")
            self.assertEqual((pr["draft"], pr["head_repo"], pr["head_sha"], pr["head_ref"]), (False, None, None, None))
            self.assertEqual((pr["base_repo"], pr["labels"], pr["author"]), ("o/r", ["a"], "u"))
            self.assertEqual([k["body"] for k in am.hae_kommentit("o/r", "1")], ["b", "c"])
            self.assertEqual(len(am.hae_tiedostot("o/r", "1")), 2)

    def test_hae_pr_virhe(self):
        with mock.patch.object(am.subprocess, "run", lambda a, **kw: subprocess.CompletedProcess(a, 1, "", "ei")):
            with self.assertRaises(RuntimeError):
                am.hae_pr("o/r", "1")

    def test_main_rikkinainen_json_ei_tee_mitaan(self):
        kutsut = []

        def f(args, **kw):
            kutsut.append(args)
            return subprocess.CompletedProcess(args, 0, "ei jsonia", "")
        with mock.patch.dict("os.environ", {"GITHUB_REPOSITORY": "o/r", "PR_NUMBER": "7"}), \
                mock.patch.object(am.subprocess, "run", f), mock.patch("builtins.print") as tuloste:
            self.assertEqual(am.main(), 1)
        self.assertIn("arviointi epaonnistui", tuloste.call_args[0][0])
        self.assertFalse(any("merge" in k for k in kutsut))

    def test_yhteenveto_ilman_ymparistomuuttujaa_vain_tulostaa(self):
        with mock.patch.dict("os.environ", clear=True), mock.patch("builtins.print") as p, \
                mock.patch("builtins.open") as avaa:
            am.yhteenveto("x")
        p.assert_called_once_with("x")
        avaa.assert_not_called()


if __name__ == "__main__":
    unittest.main()
