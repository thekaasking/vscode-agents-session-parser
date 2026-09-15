"""
__main__.py — CLI entry point for session-parser.

Invoked via:
    python -m session_parser <path> [options]
    session-parser <path> [options]       # after `uv tool install` / `pip install`
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .parser import parse_session
from .writers import write_output_dir


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog='session-parser',
        description=(
            'Parse a GitHub Copilot agent session debug log (zip or extracted directory) '
            'into a structured JSON summary.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
output modes (mutually exclusive):
  --output-dir DIR    Write stats.json + Markdown artifacts + CSVs to DIR/
  --out FILE          Write JSON to FILE (single-file mode)
  (default)           Pretty or compact JSON to stdout

examples:
  session-parser session.zip --output-dir ./out/my-session/
  session-parser session.zip --out stats.json
  session-parser session.zip --pretty
  session-parser extracted/a6ca3ad3-.../ --output-dir ./out/ --baseline
  python -m session_parser session.zip --pretty
        """,
    )
    ap.add_argument(
        'path',
        help='Path to a .zip debug-log file or a pre-extracted session directory.',
    )

    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        '--output-dir', '-d',
        metavar='DIR',
        help=(
            'Write all outputs (stats.json, prompt.md, final_response.md, '
            'artifact files, tool_calls.csv, subagents.csv) to DIR.'
        ),
    )
    mode.add_argument(
        '--out', '-o',
        metavar='FILE',
        help='Write JSON output to FILE instead of stdout.',
    )

    ap.add_argument(
        '--pretty',
        action='store_true',
        help='Pretty-print JSON (stdout or --out mode only; ignored with --output-dir).',
    )
    ap.add_argument(
        '--baseline',
        action='store_true',
        help='Mark this session as a baseline run in _meta.is_baseline.',
    )
    return ap


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()

    path = Path(args.path)
    if not path.exists():
        ap.error(f'Path does not exist: {path}')

    try:
        result = parse_session(path)
    except ValueError as exc:
        ap.error(str(exc))
        return

    if args.baseline:
        result['_meta']['is_baseline'] = True

    if args.output_dir:
        out_dir = Path(args.output_dir)
        write_output_dir(result, out_dir)
        print(f'Written to {out_dir}', file=sys.stderr)

    elif args.out:
        out_path = Path(args.out)
        indent = 2 if args.pretty else None
        out_path.write_text(
            json.dumps(result, indent=indent, ensure_ascii=False),
            encoding='utf-8',
        )
        print(f'Written to {out_path}', file=sys.stderr)

    else:
        indent = 2 if args.pretty else None
        print(json.dumps(result, indent=indent, ensure_ascii=False))


if __name__ == '__main__':
    main()
