"""
helpers.py — Pure utility functions for session parsing.

No I/O, no side effects.  All functions are stateless and independently testable.
"""

from __future__ import annotations

import json
import re
from typing import Any


# JSON helpers
def jloads(s: str | None) -> Any:
    """
    Parse a double-encoded JSON string.

    Many fields in the zip JSONL format (inputMessages, requestOptions,
    agent_response.response, tool_call.args) are JSON strings that must be
    parsed a second time.  Returns the raw value unchanged on failure.
    """
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception as e:  # noqa: BLE001
        print(f"Error parsing JSON: {e}")
        return s


def ms_to_s(ms: float) -> float:
    """Convert milliseconds to seconds, rounded to 3 decimal places."""
    return round(ms / 1000, 3)


# Discovery parsing
def parse_discovery_list(details: str) -> list[str]:
    """
    Extract the list of loaded names from a discovery event's detail string.

    Example input:
        "Resolved 7 agents in 622.0ms | loaded: [talktoys-codesearch, Plan, Ask]"
    Returns:
        ["talktoys-codesearch", "Plan", "Ask"]
    """
    m = re.search(r"loaded:\s*\[([^\]]*)\]", details)
    if not m:
        return []
    return [x.strip() for x in m.group(1).split(",") if x.strip()]


# Artifact extraction
def extract_artifacts(tool_events: list[dict]) -> list[dict]:
    """
    Detect files written by the agent from tool call events.

    Recognised sources:

    ``create_file``
        ``args.filePath`` / ``args.path``

    ``apply_patch``
        Patch header lines of the form::

            *** Add File: <path>
            *** Update File: <path>
            *** Delete File: <path>

        Falls back to unified-diff ``+++ b/<path>`` lines if the above are absent.

    Error events are skipped.  Duplicate paths are deduplicated (first occurrence wins).

    Returns a list of ``{"path": str, "operation": str}`` dicts.
    """
    artifacts: list[dict] = []
    seen: set[str] = set()

    for e in tool_events:
        name = e.get("name", "")
        if e.get("status") == "error":
            continue
        args = jloads(e.get("attrs", {}).get("args"))
        if not isinstance(args, dict):
            continue

        paths: list[str] = []

        if name == "create_file":
            p = args.get("filePath") or args.get("path")
            if p:
                paths = [p]

        elif name == "apply_patch":
            patch_input = args.get("input", "")
            paths = re.findall(
                r"^\*\*\*\s+(?:Add|Update|Delete)\s+File:\s+(.+)$",
                patch_input,
                re.MULTILINE,
            )
            if not paths:
                # Fallback: unified diff +++ b/<path>
                paths = re.findall(r"^[+]{3}\s+b?/(.+)$", patch_input, re.MULTILINE)

        for p in paths:
            p = p.strip().replace("\\", "/")
            if p and p not in seen:
                seen.add(p)
                artifacts.append({"path": p, "operation": name})

    return artifacts


# Artifact reconstruction
def _apply_hunk(base: str, diff_body: str) -> str:
    """
    Apply one or more diff hunks to *base* text.

    The patch format used by Copilot agents uses bare ``@@`` markers with no
    line-range info.  Each ``@@`` section contains a mix of:

    * ``-`` lines  — remove from base
    * ``+`` lines  — insert into result
    * `` `` lines  — context (advance cursor through base without change)

    Operations are applied sequentially using a cursor into the base lines.
    Context lines advance the cursor and are passed through unchanged.
    Removed lines must match at the cursor position and are dropped.
    Added lines are inserted at the current output position.

    Falls back to appending a ``<details>`` annotation if anchoring fails.
    """
    base_lines = base.splitlines()

    # Split on @@ hunk markers (bare "@@" or standard "@@...@@").
    hunk_bodies = re.split(r"^@@[^\n]*", diff_body, flags=re.MULTILINE)

    result: list[str] = []
    base_idx = 0  # cursor into base_lines

    for hunk in hunk_bodies:
        hunk_lines = hunk.splitlines()

        # Skip preamble/trailer lines (*** Begin Patch, *** End Patch, etc.)
        op_lines = [
            l
            for l in hunk_lines
            if l.startswith(("-", "+", " ")) and not l.startswith(("---", "+++"))
        ]
        if not op_lines:
            continue

        # Find the anchor in base for this hunk (first - or context line).
        anchor_content = None
        for l in op_lines:
            if l.startswith(("-", " ")):
                anchor_content = l[1:]
                break

        if anchor_content is not None:
            # Advance base_idx until we reach the anchor line.
            # Lines we pass over are emitted unchanged.
            found = False
            for i in range(base_idx, len(base_lines)):
                if base_lines[i] == anchor_content:
                    # Emit everything before this anchor
                    result.extend(base_lines[base_idx:i])
                    base_idx = i
                    found = True
                    break
            if not found:
                # Anchor not found — append remaining base and diff annotation.
                result.extend(base_lines[base_idx:])
                annotation = (
                    "\n\n<details>\n<summary>patch applied (approximate diff)</summary>\n\n"
                    "```diff\n" + hunk.strip() + "\n```\n</details>\n"
                )
                result.append(annotation)
                base_idx = len(base_lines)
                continue

        # Process each op line sequentially.
        for l in op_lines:
            if l.startswith("-") and not l.startswith("---"):
                # Remove: skip the matching base line.
                removed = l[1:]
                if base_idx < len(base_lines) and base_lines[base_idx] == removed:
                    base_idx += 1
                # If mismatch, skip silently (diverged).
            elif l.startswith("+") and not l.startswith("+++"):
                # Insert: emit the added line.
                result.append(l[1:])
            elif l.startswith(" "):
                # Context: pass through from base.
                ctx = l[1:]
                if base_idx < len(base_lines) and base_lines[base_idx] == ctx:
                    result.append(base_lines[base_idx])
                    base_idx += 1
                else:
                    # Context mismatch — emit as-is and do not advance cursor.
                    result.append(ctx)

    # Emit any remaining base lines after the last hunk.
    result.extend(base_lines[base_idx:])
    return "\n".join(result)


def reconstruct_artifacts(tool_events: list[dict]) -> dict[str, str | None]:
    """
    Reconstruct the final content of every file written by the agent.

    Operates on all tool call events from all agents (main + subagents), sorted
    by ``ts`` (timestamp).  Returns a dict mapping normalised file path to final
    content string, or ``None`` if the file was deleted.

    Supported tool operations:

    ``create_file``
        ``args.filePath`` / ``args.path`` + ``args.content`` → sets file content.

    ``apply_patch`` with ``*** Add File: <path>``
        Extracts lines prefixed ``+`` (stripping the leading ``+``), joins with
        newlines → sets file content.

    ``apply_patch`` with ``*** Update File: <path>``
        Appends the raw patch hunk to the existing content inside a Markdown
        ``<details>`` block.  This is an intentional approximation: the
        reconstructed file is a complete readable base (from the ``Add``
        operation) with subsequent edits annotated inline.  Sufficient for human
        review; not a byte-accurate replay.

    ``apply_patch`` with ``*** Delete File: <path>``
        Sets content to ``None``.

    Error events (``status == "error"``) are skipped.
    """
    # Sort by ts so operations are applied in chronological order.
    sorted_events = sorted(
        (e for e in tool_events if e.get("name") in ("create_file", "apply_patch")),
        key=lambda e: e.get("ts", 0),
    )

    def _norm(p: str) -> str:
        """Normalise path separators so backslash and forward-slash keys merge."""
        return p.replace("\\", "/")

    contents: dict[str, str | None] = {}

    for e in sorted_events:
        if e.get("status") == "error":
            continue
        name = e.get("name", "")
        args_raw = e.get("attrs", {}).get("args", "")
        args = jloads(args_raw)

        if name == "create_file":
            if isinstance(args, dict):
                path = _norm((args.get("filePath") or args.get("path") or "").strip())
                content = args.get("content", "")
            else:
                # args is a truncated string (debug log cuts off long content values).
                # Extract filePath and whatever content is available via regex.
                args_str = args_raw if isinstance(args_raw, str) else str(args or "")
                fp_m = re.search(
                    r'"(?:filePath|path)"\s*:\s*"((?:[^"\\]|\\.)*)"', args_str
                )
                path = _norm(fp_m.group(1).replace("\\\\", "\\")) if fp_m else ""
                # Extract content up to the truncation point — strip trailing partial escape
                ct_m = re.search(r'"content"\s*:\s*"(.*)', args_str, re.DOTALL)
                if ct_m:
                    raw = ct_m.group(1).rstrip("\\")
                    content = (
                        raw.replace("\\n", "\n")
                        .replace("\\t", "\t")
                        .replace('\\"', '"')
                        .replace("\\\\", "\\")
                    )
                    content += "\n\n*(content truncated in debug log)*"
                else:
                    content = ""
            if path:
                contents[path] = content

        elif name == "apply_patch":
            if not isinstance(args, dict):
                # Truncated args: extract the input patch via regex from the raw string.
                args_str = args_raw if isinstance(args_raw, str) else str(args or "")
                input_m = re.search(r'"input"\s*:\s*"(.*)', args_str, re.DOTALL)
                if not input_m:
                    continue
                # Unescape the truncated patch body
                raw_patch = (
                    input_m.group(1)
                    .replace("\\n", "\n")
                    .replace("\\t", "\t")
                    .replace('\\"', '"')
                    .replace("\\\\", "\\")
                )
                patch = raw_patch
            else:
                patch = args.get("input", "")
            # Find all file operations in order within this patch.
            # Format: *** <Op> File: <path>\n<lines>\n*** (next op or End Patch)
            segments = re.split(
                r"^\*\*\*\s+(Add|Update|Delete)\s+File:\s+(.+)$",
                patch,
                flags=re.MULTILINE,
            )
            # segments = [preamble, op, path, body, op, path, body, ...]
            i = 1
            while i + 2 <= len(segments):
                op = segments[i].strip()
                path = _norm(segments[i + 1].strip())
                body = segments[i + 2] if i + 2 < len(segments) else ""
                i += 3

                if op == "Delete":
                    contents[path] = None
                elif op == "Add":
                    # Extract lines: those starting with + (not +++), strip the +.
                    added = "\n".join(
                        line[1:]
                        for line in body.splitlines()
                        if line.startswith("+") and not line.startswith("+++")
                    )
                    contents[path] = added
                elif op == "Update":
                    existing = contents.get(path)
                    if existing is not None:
                        # We have the base — apply the hunk properly:
                        # keep context lines (leading space) and added lines (+),
                        # skip removed lines (-).  This gives the post-patch state.
                        contents[path] = _apply_hunk(existing, body)
                    else:
                        # No base (file was created outside this session).
                        # Reconstruct "after" state from the diff directly:
                        # context lines (space prefix) + added lines (+).
                        after_lines = []
                        for line in body.splitlines():
                            if (
                                line.startswith((" ", "\t"))
                                or line.startswith("+")
                                and not line.startswith("+++")
                            ):
                                after_lines.append(line[1:])
                            # skip - lines and @@ markers
                        contents[path] = "\n".join(after_lines)

    return contents


# Terminal command classification

# diaknose subcommands we recognise as distinct operations.
_DIAKNOSE_SUBCMDS: frozenset[str] = frozenset(
    {
        "info",
        "analyze",
        "report",
        "timeline",
        "exceptions",
        "help",
        "pre-extract",
        "version",
    }
)

# Shell keywords / noise tokens that are never the "main" binary of a step.
_SHELL_NOISE: frozenset[str] = frozenset(
    {
        "echo",
        "set",
        "cd",
        "export",
        "local",
        "if",
        "then",
        "else",
        "elif",
        "fi",
        "for",
        "do",
        "done",
        "while",
        "case",
        "esac",
        "printf",
        "#",
        "!",
        "not",
        "test",
        "true",
        "false",
    }
)

# A valid binary name: starts with a letter or underscore, contains only
# word chars, hyphens, dots, or @ (covers things like rg, rg.exe,
# Set-Location, Get-ChildItem, diaknose, pre-extract, python3, bash).
# Explicitly rejects tokens that start with $, (, ), {, }, [, " etc.
_BINARY_RE = re.compile(r"^[A-Za-z_][\w.\-@]*$")

# File extensions that are data files, not executables — filter these out.
_DATA_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".cs",
        ".xml",
        ".json",
        ".md",
        ".txt",
        ".log",
        ".ps1",
        ".yaml",
        ".yml",
        ".toml",
        ".cfg",
        ".ini",
        ".csv",
        ".ts",
        ".js",
        ".py",
        ".java",
        ".cpp",
        ".c",
        ".h",
        ".hpp",
        ".csproj",
        ".sln",
        ".targets",
        ".props",
        ".bat",
        ".sh",
        ".zip",
        ".tar",
        ".gz",
    }
)


def _is_valid_binary(tok: str) -> bool:
    """
    True if *tok* looks like an executable name, not a shell fragment or data file.

    Accepts:  rg, rg.exe, Set-Location, Get-Content, diaknose, python3, bash
    Rejects:  file.cs, config.json, '...', partial expressions
    """
    if not _BINARY_RE.match(tok):
        return False
    # Reject tokens whose extension is clearly a data file.
    dot_idx = tok.rfind(".")
    if dot_idx > 0:
        ext = tok[dot_idx:].lower()
        if ext in _DATA_EXTENSIONS:
            return False
    return True


def _shell_steps_quoted(cmd: str) -> list[str]:
    """
    Quote-aware shell command splitter.

    Splits on ``&&``, ``||``, ``;``, and `` | `` (pipe with surrounding spaces)
    only when *not* inside a single-quoted or double-quoted string.
    This prevents splitting inside rg/grep patterns like ``"foo|bar"`` or
    PowerShell for-loop bodies like ``for($i=0; $i -le 10; $i++){...}``.
    """
    steps: list[str] = []
    buf: list[str] = []
    in_single = False
    in_double = False
    i = 0
    n = len(cmd)

    def flush() -> None:
        step = "".join(buf).strip().lstrip("(").strip()
        if step:
            steps.append(step)
        buf.clear()

    while i < n:
        ch = cmd[i]

        if ch == "'" and not in_double:
            in_single = not in_single
            buf.append(ch)
            i += 1
        elif ch == '"' and not in_single:
            in_double = not in_double
            buf.append(ch)
            i += 1
        elif not in_single and not in_double:
            rest = cmd[i:]
            if rest.startswith(("&&", "||")):
                flush()
                i += 2
            elif ch == ";":
                flush()
                i += 1
            elif (
                ch == "|"
                and i > 0
                and cmd[i - 1] == " "
                and i + 1 < n
                and cmd[i + 1] == " "
            ):
                # Shell pipe: space | space
                flush()
                i += 1
            else:
                buf.append(ch)
                i += 1
        else:
            buf.append(ch)
            i += 1

    flush()
    return steps


def _extract_inner(cmd: str) -> str:
    """
    Extract the inner command string from ``docker exec diaknose bash -c "..."``
    (or ``-lc``).  Falls back to stripping the docker prefix.
    """
    # Match both single and double quoted inner strings.
    m = re.search(
        r'docker exec diaknose bash -[lc]{1,2}\s+(["\'])(.+?)\1\s*(?:2>|$)',
        cmd,
        re.DOTALL,
    )
    if m:
        return m.group(2)
    # Direct: docker exec diaknose <cmd ...>
    return re.sub(r"^.*?docker exec diaknose\s+", "", cmd)


def _first_binary(step: str) -> str:
    """
    Return the executable name from the first token of a shell step, or ``""``
    if no valid binary token is found.

    Handles:
    - Leading env-var assignments (``KEY=value``) — skipped
    - PowerShell variable assignments (``$var=Command``) — the RHS is extracted
    - Path prefixes (``/usr/bin/rg`` → ``rg``)
    """
    for tok in step.split():
        if tok.startswith("$") and "=" in tok:
            # PowerShell: $var=Command — extract what follows the =
            rhs = tok.split("=", 1)[1]
            name = rhs.replace("\\", "/").split("/")[-1].strip("'\"")
            if name and _is_valid_binary(name):
                return name
            continue
        if "=" in tok and not tok.startswith("-"):
            # Shell env assignment KEY=value — skip
            continue
        # Strip path prefix and quotes
        name = tok.replace("\\", "/").split("/")[-1].strip("'\"")
        if _is_valid_binary(name):
            return name
    return ""


def classify_terminal_command(cmd: str) -> list[dict]:
    """
    Classify a single ``run_in_terminal`` command string into structured records.

    Returns a list of dicts (one per logical sub-command), each with:

    ``category``
        ``"diaknose"`` | ``"container_other"`` | ``"host"``

    ``binary``
        Executable name (e.g. ``"rg"``, ``"jq"``, ``"diaknose"``).

    ``diaknose_subcmd``
        Diaknose subcommand for ``category == "diaknose"`` (e.g. ``"info"``).
        Empty string otherwise.

    ``flags``
        List of flag tokens (starting with ``-``) found in the step.
    """
    results: list[dict] = []
    in_container = "docker exec diaknose" in cmd
    inner = _extract_inner(cmd) if in_container else cmd

    for step in _shell_steps_quoted(inner):
        tok = _first_binary(step)
        if not tok or tok in _SHELL_NOISE:
            continue

        parts = step.split()
        flags = [p for p in parts[1:] if p.startswith("-")]

        if in_container:
            if tok == "diaknose":
                subcmd = (
                    parts[1] if len(parts) > 1 and not parts[1].startswith("-") else ""
                )
                if subcmd in _DIAKNOSE_SUBCMDS:
                    results.append(
                        {
                            "category": "diaknose",
                            "binary": "diaknose",
                            "diaknose_subcmd": subcmd,
                            "flags": flags,
                        }
                    )
                # Unrecognised diaknose token = false parse from splitting — skip.
            else:
                results.append(
                    {
                        "category": "container_other",
                        "binary": tok,
                        "diaknose_subcmd": "",
                        "flags": flags,
                    }
                )
        else:
            results.append(
                {
                    "category": "host",
                    "binary": tok,
                    "diaknose_subcmd": "",
                    "flags": flags,
                }
            )

    return results


def parse_terminal_commands(terminal_commands: list[str]) -> list[dict]:
    """
    Parse a list of raw terminal command strings into structured classification
    records.  Each record has ``category``, ``binary``, ``diaknose_subcmd``,
    and ``flags``.

    Multiple sub-commands within one chained shell invocation each produce their
    own record.
    """
    records: list[dict] = []
    for cmd in terminal_commands:
        records.extend(classify_terminal_command(cmd))
    return records


# Response text extraction
def parse_response_text(response_json_str: str) -> tuple[str, list[str]]:
    """
    Extract text and tool-call names from an ``agent_response.attrs.response`` string.

    The field is a double-encoded JSON array:
    ``[{"role": "assistant", "parts": [{type, content|name|id}, ...]}]``

    Returns:
        (joined_text, list_of_tool_call_names)
    """
    parsed = jloads(response_json_str)
    if not isinstance(parsed, list) or not parsed:
        return "", []
    parts = parsed[0].get("parts", [])
    text = "\n".join(p.get("content", "") for p in parts if p.get("type") == "text")
    tools = [
        p.get("name", p.get("id", "")) for p in parts if p.get("type") == "tool_call"
    ]
    return text, tools
