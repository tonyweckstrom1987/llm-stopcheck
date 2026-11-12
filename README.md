# llm-stopcheck

A tiny, dependency-free Python library and CLI that catches *silent* LLM failures: a response with
HTTP 200 that is cut off at `max_tokens`, empty because a reasoning model spent its whole budget
thinking, content-filtered, or missing a finish reason. It returns a stable error code and a retry hint
instead of letting half an answer flow downstream.

## Why

An LLM call can succeed (HTTP 200) and still be unusable. A real case: a reasoning model used its
whole `max_tokens` budget on thinking, `content` was empty and `finish_reason` was `length`. A daily
report was sent without its summary, and nothing said why. `llm-stopcheck` turns a response into a
small verdict (`Verdict`) whose code is stable, so you can log it and alert on it.

## Installation

```
pip install llm-stopcheck   # from PyPI (0.1.0); no dependencies, Python 3.10+
pip install .               # or from a clone of this repo, run in the repo root
```

The PyPI release 0.1.0 does not yet include the CLI options `--near-limit`, `--allow-empty` and
`--version`, nor several input files in one call (see the CLI section). They are on `main` and need a
newer release or an install from the repo. Without installing, you can run the CLI from a clone with
`PYTHONPATH=src python3 -m llm_stopcheck response.json`.

## Usage

`client` and `log` in this example are your own: any OpenAI-compatible client and any logger.

```python
import llm_stopcheck as sc

response = client.chat.completions.create(..., max_tokens=3000)

verdict = sc.check(response, max_tokens=3000)   # never raises; bad input gives malformed_response
if not verdict.ok:
    log.error(sc.format_record(verdict, model="my-model"))
    if verdict.hint == sc.HINT_RAISE_MAX_TOKENS:
        ...  # retry with a higher output cap

text = sc.ensure(response)   # returns the text, or raises TruncatedError / EmptyOutputError / ...
```

Retry with a higher output cap when `check()` says `raise_max_tokens` (`call` is your own function that
sends the request with the given `max_tokens` and returns the response):

```python
import llm_stopcheck as sc

def complete_with_retry(call, max_tokens=1000, attempts=3):
    for attempt in range(attempts):
        response = call(max_tokens=max_tokens)
        verdict = sc.check(response, max_tokens=max_tokens)
        if verdict.hint != sc.HINT_RAISE_MAX_TOKENS or attempt == attempts - 1:
            break
        max_tokens *= 2   # truncated: try again with a doubled cap
    return sc.ensure(response, max_tokens=max_tokens)   # text, or a typed error if still failing
```

`response` is parsed OpenAI-compatible JSON (OpenAI, LiteLLM, OpenRouter, vLLM, Ollama, DeepSeek) or an
object with `model_dump()`, `to_dict()` or `dict()`. `native_finish_reason` is checked too, because
LiteLLM and OpenRouter can normalise a `MAX_TOKENS` stop to `stop`.

## Codes

| code | meaning | hint |
|---|---|---|
| `ok` | fine (warnings `near_limit` / `at_limit` when `max_tokens` is given) | - |
| `truncated` | output cap reached, the answer is cut off | `raise_max_tokens` |
| `truncated_empty` | output cap reached with empty content | `raise_max_tokens` |
| `reasoning_exhausted` | the budget went to reasoning, no answer | `raise_max_tokens` |
| `tool_call_truncated` | tool-call arguments are incomplete | `raise_max_tokens` |
| `empty_output` | normal stop but empty content | `retry` |
| `content_filtered` | blocked by a provider filter | `fail` |
| `refusal` | the model refused | `fail` |
| `missing_finish_reason` | no finish reason (cut off?) | `retry` |
| `unknown_finish_reason` | unrecognised value | `retry` |
| `malformed_response` | not the expected shape | `retry` |

Content made up only of whitespace counts as empty.

`ensure()` raises a subclass of `StopcheckError` (the full `Verdict` is in `.verdict`, the code in
`.code`, the hint in `.hint`):

| exception | codes |
|---|---|
| `TruncatedError` | `truncated`, `truncated_empty`, `reasoning_exhausted`, `tool_call_truncated` |
| `EmptyOutputError` | `empty_output` |
| `ContentFilteredError` | `content_filtered` |
| `RefusalError` | `refusal` |
| `ProtocolError` | `missing_finish_reason`, `unknown_finish_reason`, `malformed_response` |

`check()` and `ensure()` also take `near_limit` (default 0.95, the `near_limit` warning threshold) and
`require_text=False` (accept empty content, for tool-call-only responses), like the CLI options of the
same purpose.

## CLI

```
llm-stopcheck response.json --max-tokens 3000
curl ... | llm-stopcheck - --json
```

Output (one line per file):
`stopcheck=reasoning_exhausted finish=length native=- tokens=3000/3000 reasoning_tokens=3000 hint=raise_max_tokens`.
Exit code 0 = ok, 1 = failed verdict, 2 = usage error, unreadable input or an internal error (an
unexpected exception or an unreadable file prints the path as you gave it and only the exception type
name to stderr, never the exception text, so it is never mistaken for a failed verdict). Warnings (`near_limit`,
`at_limit`) do not change the exit code and are printed to stderr.

Several files may be given (`llm-stopcheck a.json b.json`): each gets its own output line ending in
`file=<path>` (a `"file"` key with `--json`), and the exit code is the worst of them (not in PyPI 0.1.0).
In the text line, characters other than `A-Z a-z 0-9 _ . : / + -` in the path become `_` (so a space
cannot split the line); `--json` keeps the exact path.
A single file prints without the name. Options: `--near-limit 0.9` changes the `near_limit`
warning threshold (default 0.95), `--allow-empty` accepts empty content (tool-call-only responses), and
`--version` prints the version. These three also need a newer version than PyPI 0.1.0; one file,
`--max-tokens` and `--json` work in 0.1.0.

`format_record` sanitises its values (ASCII only, no newlines, `=` replaced, at most 60 characters per
value), so model output cannot forge a log record.

## Limitations

- Only the first choice (`choices[0]`) is checked; other choices are ignored.
- The input is a complete, parsed response in the OpenAI chat-completion shape (`choices[].message`).
  Streams are not consumed, and a response without `choices` (for example a native Anthropic Messages
  body) gives `malformed_response`.
- The verdict rests on the finish reason, the content and the usage counts. Answer quality is not judged,
  and tool-call arguments are not parsed: they are flagged only when the finish reason says the cap was hit.
- An unrecognised finish reason gives `unknown_finish_reason`.
- `near_limit` / `at_limit` need both `max_tokens` and a completion token count in `usage`, and only
  appear on responses that are otherwise ok.
- The library does not call a provider and does not alert. Retrying and alerting are up to the caller.

## Alternatives and why this one

This check usually exists as framework-specific fixes or blog recipes, not as a small standalone
library. This one is deliberately small: no dependencies, no network, no framework coupling. Alerting
(Telegram, healthchecks.io) is left to the caller; `format_record` and the exit codes are enough to hook
it up.

## Tests

```
python -m unittest discover -s tests -t .
```

## License

MIT
