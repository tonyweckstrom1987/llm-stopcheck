"""Command line: check a saved response (file or stdin) and set the exit code.

    llm-stopcheck response.json --max-tokens 3000
    llm-stopcheck a.json b.json --near-limit 0.9 --allow-empty
    curl ... | llm-stopcheck - --json

Several files give one output line each, with the file name added (file=...); the exit code is the
worst of them. A single file prints exactly as before.
Exit codes: 0 ok (warnings go to stderr), 1 failed verdict, 2 usage error, unreadable input or an
internal error (only the exception type name is printed, never the exception text).
"""
from __future__ import annotations

import argparse
import json
import sys

from . import __version__, core, errors


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="llm-stopcheck", description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="+", metavar="path", help="response JSON file(s), or - for stdin")
    parser.add_argument("--max-tokens", type=int, default=None, help="output cap that was sent")
    parser.add_argument("--near-limit", type=float, default=0.95,
                        help="fraction of --max-tokens that triggers the near_limit warning (default 0.95)")
    parser.add_argument("--allow-empty", action="store_true",
                        help="do not fail on empty content (e.g. tool-call-only responses)")
    parser.add_argument("--json", action="store_true", dest="as_json", help="print the verdict as JSON")
    parser.add_argument("--version", action="version", version="%(prog)s " + __version__)
    try:
        args = parser.parse_args(argv)
    except SystemExit as exit_:
        return 2 if exit_.code else 0
    if args.max_tokens is not None and args.max_tokens <= 0:
        print("llm-stopcheck: --max-tokens must be positive", file=sys.stderr)
        return 2
    if not 0 < args.near_limit <= 1:
        print("llm-stopcheck: --near-limit must be in (0, 1]", file=sys.stderr)
        return 2
    show_name = len(args.paths) > 1
    return max(_check_guarded(path, args, show_name) for path in args.paths)


def _check_guarded(path, args, show_name) -> int:
    """Turn an unexpected exception into exit code 2, so it cannot pass for a failed verdict (1)."""
    try:
        return _check_one(path, args, show_name)
    except Exception as exc:  # noqa: BLE001 - last line of defence; the message may hold secrets
        print("llm-stopcheck: internal error on %s: %s" % (path, type(exc).__name__), file=sys.stderr)
        return 2


def _check_one(path, args, show_name=False) -> int:
    try:
        if path == "-":
            data = json.load(sys.stdin)
        else:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
    except (OSError, ValueError) as exc:
        print("llm-stopcheck: cannot read input %s: %s" % (path, type(exc).__name__), file=sys.stderr)
        return 2
    verdict = core.check(data, max_tokens=args.max_tokens, near_limit=args.near_limit,
                         require_text=not args.allow_empty)
    if args.as_json:
        record = verdict.as_dict()
        if show_name:
            record["file"] = path
        print(json.dumps(record, sort_keys=True))
    elif show_name:
        # Not format_record(file=...): its values are cut to 60 characters, which would drop the
        # file name from a long path. The path is sanitised the same way (one key=value token).
        print(errors.format_record(verdict) + " file=" + _file_value(path))
    else:
        print(errors.format_record(verdict))
    if verdict.warnings:
        print("warnings: " + ", ".join(verdict.warnings), file=sys.stderr)
    return 0 if verdict.ok else 1


def _file_value(path) -> str:
    """The path as one log-safe token: characters outside [A-Za-z0-9_.:/+-] become _, no length cut."""
    return errors._SAFE.sub("_", str(path)) or "-"


if __name__ == "__main__":
    sys.exit(main())
