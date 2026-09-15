"""
writers.py — Output writers for parsed session data.

Provides :func:`write_output_dir`, which takes a parsed session dict (as
returned by :func:`~session_parser.parser.parse_session`) and writes a
self-contained output directory containing:

* ``stats.json``              — full parse result, pretty-printed
* ``prompt.md``               — initial user prompt
* ``final_response.md``       — last text-only agent response
* ``<basename>.md``           — one file per reconstructed agent-created artifact
* ``tool_calls.csv``          — per (agent x tool_name) statistics
* ``subagents.csv``           — per subagent invocation
* ``terminal_commands.csv``   — per-binary breakdown of run_in_terminal calls,
                                split into host / container_other / diaknose

All output is generic and session-specific only in content — no experiment
labels or cross-session context are embedded here.
"""

from __future__ import annotations

import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

from .helpers import parse_terminal_commands

# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────


def write_output_dir(session: dict, output_dir: Path) -> None:
    """
    Write all output files for one parsed session into *output_dir*.

    Creates the directory if it does not exist.  Existing files are overwritten.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_stats_json(session, output_dir)
    _write_prompt_md(session, output_dir)
    _write_final_response_md(session, output_dir)
    _write_artifact_files(session, output_dir)
    _write_tool_calls_csv(session, output_dir)
    _write_tool_errors_csv(session, output_dir)
    _write_subagents_csv(session, output_dir)
    _write_terminal_commands_csv(session, output_dir)


# ──────────────────────────────────────────────────────────────────────────────
# Markdown / JSON writers
# ──────────────────────────────────────────────────────────────────────────────


def _write_stats_json(session: dict, out: Path) -> None:
    (out / "stats.json").write_text(
        json.dumps(session, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_prompt_md(session: dict, out: Path) -> None:
    prompt = session.get("interaction", {}).get("initial_prompt") or ""
    (out / "prompt.md").write_text(prompt, encoding="utf-8")


def _write_final_response_md(session: dict, out: Path) -> None:
    response = session.get("interaction", {}).get("final_answer") or ""
    (out / "final_response.md").write_text(response, encoding="utf-8")


def _write_artifact_files(session: dict, out: Path) -> None:
    """Write each reconstructed artifact to a sanitised filename."""
    contents: dict[str, str | None] = session.get("artifact_contents", {})
    for path, content in contents.items():
        if content is None:
            continue
        filename = _sanitise_filename(Path(path).name)
        (out / filename).write_text(content, encoding="utf-8")


def _sanitise_filename(name: str) -> str:
    """Make a path basename safe for the local filesystem."""
    name = re.sub(r"[^\w.\-]", "_", name)
    return name or "artifact"


# ──────────────────────────────────────────────────────────────────────────────
# tool_calls.csv
# ──────────────────────────────────────────────────────────────────────────────

_TOOL_CALLS_FIELDS = [
    "agent_name",
    "tool_name",
    "call_count",
    "error_count",
    "error_rate_pct",
    "total_dur_s",
    "avg_dur_s",
    "p50_dur_s",
    "p95_dur_s",
    "max_dur_s",
    "parallel_batch_count",
    "calls_in_parallel_batches",
]


def _write_tool_calls_csv(session: dict, out: Path) -> None:
    """
    One row per (agent_name × tool_name) across main agent and all subagents.

    Derives per-tool statistics from the per-turn tool_call records which carry
    ``ts_ms`` and ``dur_ms`` for every individual call.

    Parallel batch detection: calls within the same agent that share an
    identical ``ts_ms`` value were issued in a single LLM response batch.
    ``parallel_batch_count`` counts how many such batches exist;
    ``calls_in_parallel_batches`` counts the total calls involved.
    """
    rows = _collect_tool_stats(session)
    _write_csv(out / "tool_calls.csv", _TOOL_CALLS_FIELDS, rows)


def _collect_tool_stats(session: dict) -> list[dict]:
    """Collect per-(agent × tool) call records from all turns across all agents."""
    # Gather (agent_name, turn list) pairs
    agents: list[tuple[str, list[dict]]] = [
        ("main", session.get("main_turns", [])),
    ]
    for sa in session.get("subagents", []):
        sa_name = sa.get("agent_name") or sa.get("call_id", "subagent")
        agents.append((sa_name, sa.get("turns", [])))

    rows: list[dict] = []
    for agent_name, turns in agents:
        # Group raw call records by tool_name, preserving ts_ms and dur_ms.
        groups: dict[str, list[dict]] = defaultdict(list)
        for turn in turns:
            for tc in turn.get("tool_calls", []):
                groups[tc["name"]].append(tc)

        for tool_name, calls in sorted(groups.items()):
            durations_ms = [c.get("dur_ms", 0) for c in calls]
            durations_s = [d / 1000 for d in durations_ms]
            errors = sum(1 for c in calls if c.get("error"))
            n = len(calls)

            # Parallel batch detection: group by ts_ms, find groups with >1 call.
            ts_groups: dict[int | None, int] = defaultdict(int)
            for c in calls:
                ts_groups[c.get("ts_ms")] += 1
            parallel_batches = {
                ts: cnt for ts, cnt in ts_groups.items() if ts is not None and cnt > 1
            }
            parallel_batch_count = len(parallel_batches)
            calls_in_parallel = sum(parallel_batches.values())

            rows.append(
                {
                    "agent_name": agent_name,
                    "tool_name": tool_name,
                    "call_count": n,
                    "error_count": errors,
                    "error_rate_pct": _pct(errors, n),
                    "total_dur_s": _r(sum(durations_s)),
                    "avg_dur_s": _r(statistics.mean(durations_s)) if durations_s else 0,
                    "p50_dur_s": _r(statistics.median(durations_s))
                    if durations_s
                    else 0,
                    "p95_dur_s": _r(_percentile(durations_s, 95)) if durations_s else 0,
                    "max_dur_s": _r(max(durations_s)) if durations_s else 0,
                    "parallel_batch_count": parallel_batch_count,
                    "calls_in_parallel_batches": calls_in_parallel,
                }
            )

    return rows


# ──────────────────────────────────────────────────────────────────────────────
# tool_errors.csv
# ──────────────────────────────────────────────────────────────────────────────

_TOOL_ERRORS_FIELDS = [
    "agent_name",
    "tool_name",
    "error_category",  # "timeout" | "canceled" | "not_found" | "regex_error" | "other"
    "error_count",
    "example_error",  # first 120 chars of a representative error message
]

_ERROR_CATEGORIES = [
    ("timeout", lambda e: "timeout" in e.lower()),
    (
        "canceled",
        lambda e: (
            e.strip().lower() in ("canceled", "cancelled")
            or e.lower().startswith("agent error")
        ),
    ),
    (
        "not_found",
        lambda e: (
            "does not exist" in e.lower()
            or "no such file" in e.lower()
            or "cannot open" in e.lower()
        ),
    ),
    ("regex_error", lambda e: "regex" in e.lower() or "parse error" in e.lower()),
]


def _categorise_error(msg: str) -> str:
    for name, test in _ERROR_CATEGORIES:
        if test(msg):
            return name
    return "other"


def _write_tool_errors_csv(session: dict, out: Path) -> None:
    """
    One row per (agent_name × tool_name × error_category).

    Aggregates all error_list entries from the main agent and all subagents,
    classifying the error message into a coarse category:
    - ``timeout``     — search / file-access timeout
    - ``canceled``    — subagent or tool invocation was canceled
    - ``not_found``   — file or directory does not exist
    - ``regex_error`` — invalid regex pattern
    - ``other``       — anything else
    """
    # Collect (agent_name, tool_name, error_msg) triples
    raw: list[tuple[str, str, str]] = []

    for e in session.get("tool_calls", {}).get("main_agent", {}).get("error_list", []):
        raw.append(("main", e["name"], e.get("error", "")))

    for sa in session.get("subagents", []):
        sa_name = sa.get("agent_name") or sa.get("call_id", "subagent")
        tc = sa.get("tool_calls") or {}
        for e in tc.get("error_list", []):
            raw.append((sa_name, e["name"], e.get("error", "")))

    # Group by (agent, tool, category)
    groups: dict[tuple, list[str]] = defaultdict(list)
    for agent, tool, msg in raw:
        cat = _categorise_error(msg)
        groups[(agent, tool, cat)].append(msg)

    rows = []
    for (agent, tool, cat), msgs in sorted(groups.items()):
        rows.append(
            {
                "agent_name": agent,
                "tool_name": tool,
                "error_category": cat,
                "error_count": len(msgs),
                "example_error": msgs[0][:120],
            }
        )

    _write_csv(out / "tool_errors.csv", _TOOL_ERRORS_FIELDS, rows)


# ──────────────────────────────────────────────────────────────────────────────
# subagents.csv
# ──────────────────────────────────────────────────────────────────────────────

_SUBAGENTS_FIELDS = [
    "invocation_index",
    "agent_name",
    "call_id",
    "parent_turn_id",
    "dur_s",
    "tokens_input",
    "tokens_output",
    "tokens_cached",
    "llm_calls",
    "cache_hit_rate_pct",
    "tools_total",
    "tools_errors",
    "terminal_cmd_count",
    "structured_output_status",
    "structured_output_len",
    "structured_output_snippet",
]


def _write_subagents_csv(session: dict, out: Path) -> None:
    """One row per subagent invocation in call order."""
    rows: list[dict] = []
    # Track per-agent-name invocation index for readability.
    invocation_counters: dict[str, int] = defaultdict(int)

    for sa in session.get("subagents", []):
        a_name = sa.get("agent_name") or "unknown"
        invocation_counters[a_name] += 1
        idx = invocation_counters[a_name]

        tokens = sa.get("tokens") or {}
        inp = tokens.get("input_total", 0)
        out_tok = tokens.get("output_total", 0)
        cached = tokens.get("cached_total", 0)
        llm_calls = tokens.get("llm_calls", 0)

        tc = sa.get("tool_calls") or {}
        so_text = sa.get("structured_output") or ""
        so_status = _classify_structured_output(so_text)

        rows.append(
            {
                "invocation_index": idx,
                "agent_name": a_name,
                "call_id": sa.get("call_id", ""),
                "parent_turn_id": sa.get("parent_turn_id", ""),
                "dur_s": sa.get("dur_s", ""),
                "tokens_input": inp,
                "tokens_output": out_tok,
                "tokens_cached": cached,
                "llm_calls": llm_calls,
                "cache_hit_rate_pct": _pct(cached, inp) if inp else 0,
                "tools_total": tc.get("total", 0),
                "tools_errors": tc.get("errors", 0),
                "terminal_cmd_count": len(tc.get("terminal_commands", [])),
                "structured_output_status": so_status,
                "structured_output_len": len(so_text),
                "structured_output_snippet": so_text[:200].replace("\n", " "),
            }
        )

    _write_csv(out / "subagents.csv", _SUBAGENTS_FIELDS, rows)


def _classify_structured_output(text: str) -> str:
    """Classify the subagent structured output status from its text content."""
    if not text:
        return "empty"
    lower = text.lower()
    if "canceled" in lower or "cancelled" in lower:
        return "canceled"
    if lower.startswith("agent error") or "agent error:" in lower:
        return "error"
    return "ok"


# ──────────────────────────────────────────────────────────────────────────────
# terminal_commands.csv
# ──────────────────────────────────────────────────────────────────────────────

_TERMINAL_CMDS_FIELDS = [
    "agent_name",
    "category",  # "host" | "container_other" | "diaknose"
    "binary",  # executable name, e.g. "rg", "jq", "diaknose"
    "diaknose_subcmd",  # e.g. "info", "analyze", "report" — empty for non-diaknose
    "call_count",
    "top_flags",  # comma-separated top-5 most frequent flags
]


def _write_terminal_commands_csv(session: dict, out: Path) -> None:
    """
    One row per (agent_name × category × binary × diaknose_subcmd).

    Three categories:
    - ``host``             Commands run on the host system.
    - ``container_other``  Commands run inside the ``diaknose`` Docker container,
                           other than the ``diaknose`` binary itself.
    - ``diaknose``         Invocations of the ``diaknose`` binary inside the
                           container, broken down by subcommand (info / analyze /
                           report / timeline / exceptions / ...).
    """
    # Collect all terminal commands from main agent and all subagents.
    agents: list[tuple[str, list[str]]] = []
    main_tc = session.get("tool_calls", {}).get("main_agent", {})
    agents.append(("main", main_tc.get("terminal_commands", [])))
    for sa in session.get("subagents", []):
        sa_name = sa.get("agent_name") or sa.get("call_id", "subagent")
        sa_tc = sa.get("tool_calls") or {}
        agents.append((sa_name, sa_tc.get("terminal_commands", [])))

    # Aggregate: (agent_name, category, binary, subcmd) → {count, flags counter}
    agg: dict[tuple, dict] = defaultdict(
        lambda: {"count": 0, "flags": defaultdict(int)}
    )
    for agent_name, cmds in agents:
        for rec in parse_terminal_commands(cmds):
            key = (agent_name, rec["category"], rec["binary"], rec["diaknose_subcmd"])
            agg[key]["count"] += 1
            for flag in rec["flags"]:
                agg[key]["flags"][flag] += 1

    rows = []
    for (agent_name, category, binary, subcmd), data in sorted(agg.items()):
        top_flags = ", ".join(
            f for f, _ in sorted(data["flags"].items(), key=lambda x: -x[1])[:5]
        )
        rows.append(
            {
                "agent_name": agent_name,
                "category": category,
                "binary": binary,
                "diaknose_subcmd": subcmd,
                "call_count": data["count"],
                "top_flags": top_flags,
            }
        )

    _write_csv(out / "terminal_commands.csv", _TERMINAL_CMDS_FIELDS, rows)


# ──────────────────────────────────────────────────────────────────────────────
# Shared CSV utility
# ──────────────────────────────────────────────────────────────────────────────


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Numeric helpers
# ──────────────────────────────────────────────────────────────────────────────


def _r(v: float, ndigits: int = 3) -> float:
    return round(v, ndigits)


def _pct(numerator: float, denominator: float, ndigits: int = 1) -> float:
    if not denominator:
        return 0.0
    return round(100 * numerator / denominator, ndigits)


def _percentile(data: list[float], p: int) -> float:
    """Return the p-th percentile of *data* (0–100). Uses nearest-rank method."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    n = len(sorted_data)
    idx = max(0, int(p / 100 * n) - 1)
    return sorted_data[min(idx, n - 1)]
