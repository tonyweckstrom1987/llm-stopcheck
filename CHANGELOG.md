# Changelog

## Unreleased

- README: limitations sections (EN and FI), install-without-pip note, and a note that `client` / `log` in the Finnish usage example are the caller's own.
- CLI: `--near-limit`, `--allow-empty` and `--version`; several files accepted, exit code is the worst one.
- Julkaisuvalmius: project.urls ja .github/workflows/julkaise.yml (PyPI Trusted Publishing, ajetaan vain release-julkaisusta).
- README: working `complete_with_retry` example that doubles `max_tokens` while `check()` hints `raise_max_tokens`; tests run the README code with a fake call (length first, then stop).
- CI: Dependabot patch/minor updates are merged automatically (major stays for a human).
- Fix: `extract_text()` raised `AttributeError` when `choices[0].message` was not an object; it now returns an empty string like the rest of the API.
- Tests: 19 new tests for edge cases (objects with `dict()`/`to_dict()`, token-count parsing, provider-specific native finish reason, content parts, CLI stdin/warnings/`python -m`); branch coverage 91 % -> 98 %.
- README: English quick start with code table and CLI notes.

## 0.1.0

- `check()` classifies chat-completion responses: truncated, truncated_empty, reasoning_exhausted,
  tool_call_truncated, empty_output, content_filtered, refusal, missing/unknown finish reason, malformed.
- `ensure()` and typed exceptions; `format_record()` for safe one-line log records.
- `llm-stopcheck` CLI (exit 0 ok, 1 failed verdict, 2 usage).
- Standard library only, Python 3.10+.
