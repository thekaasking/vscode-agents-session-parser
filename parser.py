"""
parser.py — Core parsing logic for GitHub Copilot agent sessions.

Two public functions:

:func:`parse_agent_jsonl`
    Parses a single JSONL file (main agent or one subagent) into a structured
    dict.  Handles all event types from the zip debug-log format.

:func:`parse_session`
    Entry point for a full session.  Accepts a zip path or extracted directory,
    delegates to :func:`parse_agent_jsonl` for every JSONL file, then assembles
    the aggregated session dict with cross-agent token totals, subagent linkage,
    and timing.

──────────────────────────────────────────────────────────────────────────────
FIELD RATIONALE
──────────────────────────────────────────────────────────────────────────────
KEPT:
  session.*          Identity, timing, agent/model context.
  user_turns         Each user prompt + full agent response cycle.
  subagents[]        Per-invocation metrics: cost, time, tools, structured output.
  tokens.*           Input / output / cached — main and subagent breakdown.
  tool_calls.*       Count, types, errors, durations, executed terminal commands.
  artifacts          Files created (create_file) or patched (apply_patch).
  timing.*           Wall time per turn and subagent; total agent-active time.
  reasoning_effort   From requestOptions.reasoning.effort — low/medium/high.
  discovery          Agents/skills/instructions loaded at session start.

DISCARDED:
  inputMessages      Full conversation history per LLM call.  Megabytes per
                     codesearch subagent; reconstructable from the event stream;
                     not a scalar eval metric.
  system_prompt_0    Static per agent version — irrelevant for per-session diff.
  tools_0.json       Tool schema definitions — static metadata.
  models.json        Model catalogue — not session-specific.
  requestShape       Internal API plumbing (inputItemCount / inputItemTypes).
  copilotUsageNanoAiu  Proprietary billing unit; input+output tokens are canonical.
  traceId            Always "exported-trace" — zero information.
  spanId/parentSpanId  Internal graph keys; used for linking during parsing only.
  vscodeVersion      Not a performance signal.
  agent_response.reasoning  Always "[encrypted]" in zip files.
  discovery source/category  Filesystem path metadata — noise.
  generic events     Per-turn Resolve Customisations — instruction-matching noise.
  title subagent     Auto session-naming call; not agent work.
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .helpers import (
    extract_artifacts,
    jloads,
    ms_to_s,
    parse_discovery_list,
    parse_response_text,
    reconstruct_artifacts,
)
from .io import SessionSource

# ──────────────────────────────────────────────────────────────────────────────
# Single-file parser
# ──────────────────────────────────────────────────────────────────────────────


def parse_agent_jsonl(events: list[dict], file_label: str = "main") -> dict:
    """
    Parse all events from one JSONL file into a structured result dict.

    Works for both the main agent (``main.jsonl``) and any subagent file
    (``runSubagent-*.jsonl``).  The ``file_label`` is stored for reference only.

    Internal linking keys (prefixed ``_``) are used to join subagent calls to
    their ``runSubagent`` tool events; they are stripped by :func:`parse_session`
    before the final output is assembled.
    """
    result: dict[str, Any] = {
        "label": file_label,
        "session_id": None,
        "parent_session_id": None,
        "agent_name": None,
        "is_subagent": False,
        "model": None,
        "reasoning_effort": None,
        "timing": {
            "session_start_ms": None,
            "session_end_ms": None,
            "wall_time_s": None,
            "agent_active_s": None,
        },
        "discovery": {
            "agents": [],
            "skills": [],
            "instructions": [],
            "slash_commands": [],
        },
        "user_messages": [],
        "agent_turns": [],
        "tokens": {
            "input_total": 0,
            "output_total": 0,
            "cached_total": 0,
            "llm_calls": 0,
            "per_call": [],
        },
        "tool_calls": {
            "total": 0,
            "errors": 0,
            "by_type": {},
            "error_list": [],
            "duration_by_type": {},
            "terminal_commands": [],
        },
        "subagent_calls": [],
        "artifacts": [],
        "final_responses": [],
    }

    turns: dict[str, dict] = {}
    pending_turn: str | None = None
    all_tool_events: list[dict] = []
    ts_all: list[int] = []

    for e in events:
        etype = e.get("type", "")
        ts = e.get("ts")
        if ts:
            ts_all.append(ts)

        # ── session_start ────────────────────────────────────────────────
        if etype == "session_start":
            result["session_id"] = e.get("sid") or e.get("attrs", {}).get("sessionId")
            attrs = e.get("attrs", {})
            result["parent_session_id"] = attrs.get("parentSessionId")
            result["agent_name"] = attrs.get("label")
            result["is_subagent"] = bool(result["parent_session_id"])

        # ── discovery ───────────────────────────────────────────────────
        elif etype == "discovery":
            name = e.get("name", "")
            details = e.get("attrs", {}).get("details", "")
            loaded = parse_discovery_list(details)
            if "Agent" in name:
                result["discovery"]["agents"] = loaded
            elif "Skill" in name:
                result["discovery"]["skills"] = loaded
            elif "Instructions" in name:
                result["discovery"]["instructions"] = loaded
            elif "Slash" in name:
                result["discovery"]["slash_commands"] = loaded

        # ── user_message ────────────────────────────────────────────────
        elif etype == "user_message":
            result["user_messages"].append(e.get("attrs", {}).get("content", ""))

        # ── turn_start ───────────────────────────────────────────────────
        elif etype == "turn_start":
            tid = e.get("attrs", {}).get("turnId", "")
            turns[tid] = {
                "turn_id": tid,
                "start_ms": ts,
                "end_ms": None,
                "duration_s": None,
                "llm_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cached_tokens": 0,
                "tool_calls": [],
                "subagent_calls": [],
                "has_terminal_cmds": False,
                "response_texts": [],
            }
            pending_turn = tid

        # ── turn_end ─────────────────────────────────────────────────────
        elif etype == "turn_end":
            tid = e.get("attrs", {}).get("turnId", "")
            if tid in turns:
                turns[tid]["end_ms"] = ts
                if turns[tid]["start_ms"] and ts:
                    turns[tid]["duration_s"] = ms_to_s(ts - turns[tid]["start_ms"])

        # ── llm_request ──────────────────────────────────────────────────
        elif etype == "llm_request":
            attrs = e.get("attrs", {})
            model = attrs.get("model", "")
            if not result["model"]:
                result["model"] = model

            inp = attrs.get("inputTokens", 0) or 0
            out = attrs.get("outputTokens", 0) or 0
            cached = attrs.get("cachedTokens", 0) or 0
            ttft = attrs.get("ttft", 0) or 0
            dur = e.get("dur", 0) or 0

            if not result["reasoning_effort"]:
                opts = jloads(attrs.get("requestOptions"))
                if isinstance(opts, dict):
                    r = opts.get("reasoning", {})
                    result["reasoning_effort"] = (
                        r.get("effort") if isinstance(r, dict) else None
                    )

            result["tokens"]["input_total"] += inp
            result["tokens"]["output_total"] += out
            result["tokens"]["cached_total"] += cached
            result["tokens"]["llm_calls"] += 1
            result["tokens"]["per_call"].append(
                {
                    "model": model,
                    "turn_id": pending_turn,
                    "input": inp,
                    "output": out,
                    "cached": cached,
                    "ttft_ms": ttft,
                    "duration_ms": dur,
                }
            )

            if pending_turn and pending_turn in turns:
                t = turns[pending_turn]
                t["llm_calls"] += 1
                t["input_tokens"] += inp
                t["output_tokens"] += out
                t["cached_tokens"] += cached

        # ── tool_call ────────────────────────────────────────────────────
        elif etype == "tool_call":
            tname = e.get("name", "")
            dur = e.get("dur", 0) or 0
            attrs = e.get("attrs", {})
            is_error = e.get("status") == "error" or bool(attrs.get("error"))

            tc = result["tool_calls"]
            tc["total"] += 1
            tc["by_type"][tname] = tc["by_type"].get(tname, 0) + 1
            tc["duration_by_type"][tname] = tc["duration_by_type"].get(tname, 0) + dur

            if is_error:
                tc["errors"] += 1
                tc["error_list"].append(
                    {
                        "name": tname,
                        "error": str(attrs.get("error", ""))[:300],
                        "ts_ms": ts,
                    }
                )

            if tname == "run_in_terminal":
                args = jloads(attrs.get("args"))
                if isinstance(args, dict) and args.get("command"):
                    tc["terminal_commands"].append(args["command"])
                    if pending_turn and pending_turn in turns:
                        turns[pending_turn]["has_terminal_cmds"] = True

            all_tool_events.append(e)

            if pending_turn and pending_turn in turns:
                turns[pending_turn]["tool_calls"].append(
                    {
                        "name": tname,
                        "ts_ms": ts,
                        "dur_ms": dur,
                        "error": is_error,
                        "result_snippet": str(attrs.get("result", ""))[:200],
                    }
                )

        # ── child_session_ref ────────────────────────────────────────────
        elif etype == "child_session_ref":
            attrs = e.get("attrs", {})
            label = attrs.get("label", "")
            if label == "title":
                continue  # title sub-call is noise, not agent work
            result["subagent_calls"].append(
                {
                    "agent_name": label,
                    "call_id": attrs.get("childSessionId", ""),
                    "child_file": attrs.get("childLogFile", ""),
                    "turn_id": pending_turn,
                    # Internal key: child_session_ref.parentSpanId == runSubagent tool_call.spanId
                    "_parent_span": e.get("parentSpanId", ""),
                    "dur_s": None,
                    "structured_output": None,
                }
            )
            if pending_turn and pending_turn in turns:
                turns[pending_turn]["subagent_calls"].append(
                    attrs.get("childSessionId", "")
                )

        # ── agent_response ───────────────────────────────────────────────
        elif etype == "agent_response":
            text, tool_names = parse_response_text(
                e.get("attrs", {}).get("response", "")
            )
            is_final = bool(text) and not tool_names
            if pending_turn and pending_turn in turns and text:
                turns[pending_turn]["response_texts"].append(text)
            if is_final:
                result["final_responses"].append(
                    {"turn_id": pending_turn, "text": text}
                )

        # ── subagent span (in child JSONL; dur = wall time) ───────────────
        elif etype == "subagent":
            dur = e.get("dur", 0) or 0
            sid = e.get("sid", "")
            for ref in result["subagent_calls"]:
                if ref["call_id"] == sid and ref.get("dur_s") is None:
                    ref["dur_s"] = ms_to_s(dur)
                    break

    # ── Link runSubagent tool_call → child_session_ref via parentSpanId ───
    for e in all_tool_events:
        if e.get("name") == "runSubagent":
            span_id = e.get("spanId", "")
            dur = e.get("dur", 0) or 0
            result_text = e.get("attrs", {}).get("result", "")
            for ref in result["subagent_calls"]:
                if ref.get("_parent_span") == span_id:
                    ref["dur_s"] = ms_to_s(dur)
                    ref["structured_output"] = result_text or None
                    break

    # ── Derived collections ───────────────────────────────────────────────
    result["artifacts"] = extract_artifacts(all_tool_events)

    if ts_all:
        result["timing"]["session_start_ms"] = min(ts_all)
        result["timing"]["session_end_ms"] = max(ts_all)
        result["timing"]["wall_time_s"] = ms_to_s(max(ts_all) - min(ts_all))

    active_ms = sum(
        t["duration_s"] * 1000 for t in turns.values() if t["duration_s"] is not None
    )
    result["timing"]["agent_active_s"] = ms_to_s(active_ms)

    result["tool_calls"]["duration_by_type"] = {
        k: ms_to_s(v) for k, v in result["tool_calls"]["duration_by_type"].items()
    }

    result["agent_turns"] = [
        {
            "turn_id": t["turn_id"],
            "duration_s": t["duration_s"],
            "llm_calls": t["llm_calls"],
            "input_tokens": t["input_tokens"],
            "output_tokens": t["output_tokens"],
            "cached_tokens": t["cached_tokens"],
            "tool_calls": t["tool_calls"],
            "subagent_calls": t["subagent_calls"],
            "has_terminal_cmds": t["has_terminal_cmds"],
            "response_snippets": [r[:300] for r in t["response_texts"]],
        }
        for t in turns.values()
    ]

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Full session parser
# ──────────────────────────────────────────────────────────────────────────────


def parse_session(source_path: str | Path) -> dict:
    """
    Parse a complete Copilot agent session from a zip file or extracted directory.

    Loads ``main.jsonl`` plus every ``runSubagent-*.jsonl`` found in the source,
    aggregates tokens and tool calls across all agents, links subagent structured
    outputs back to the parent's tool calls, and returns a single flat dict
    suitable for JSON serialisation.

    Parameters
    ----------
    source_path:
        Path to a ``.zip`` file or a pre-extracted session directory.

    Returns
    -------
    dict
        Structured session summary.  See module docstring for field rationale.
    """
    with SessionSource(source_path) as src:
        # ── Main agent ────────────────────────────────────────────────────
        main_events = list(src.iter_jsonl("main.jsonl"))
        main = parse_agent_jsonl(main_events, file_label="main")

        # Strip internal linking keys — not needed in output
        for ref in main["subagent_calls"]:
            ref.pop("_parent_span", None)

        # ── Subagent files ────────────────────────────────────────────────
        subagent_details: dict[str, dict] = {}
        _sa_raw_events: dict[
            str, list[dict]
        ] = {}  # fname → raw events (for artifact recon)
        for fname in src.list_jsonl():
            if fname == "main.jsonl" or fname.startswith("title-"):
                continue
            events = list(src.iter_jsonl(fname))
            if not events:
                continue
            _sa_raw_events[fname] = events
            parsed = parse_agent_jsonl(events, file_label=fname)

            # Propagate structured_output from main's subagent_calls if richer
            for ref in main["subagent_calls"]:
                if ref["child_file"] == fname or ref["call_id"] == parsed.get(
                    "session_id"
                ):
                    if ref.get("structured_output") and not parsed.get(
                        "structured_output"
                    ):
                        parsed["structured_output"] = ref["structured_output"]
                    break

            subagent_details[fname] = parsed

        # ── Cross-agent token aggregation ─────────────────────────────────
        total_input = main["tokens"]["input_total"]
        total_output = main["tokens"]["output_total"]
        total_cached = main["tokens"]["cached_total"]
        total_llm_calls = main["tokens"]["llm_calls"]
        for sa in subagent_details.values():
            total_input += sa["tokens"]["input_total"]
            total_output += sa["tokens"]["output_total"]
            total_cached += sa["tokens"]["cached_total"]
            total_llm_calls += sa["tokens"]["llm_calls"]

        # ── Cross-agent tool call aggregation ─────────────────────────────
        agg_by_type: dict[str, int] = dict(main["tool_calls"]["by_type"])
        agg_total = main["tool_calls"]["total"]
        agg_errors = main["tool_calls"]["errors"]
        for sa in subagent_details.values():
            for k, v in sa["tool_calls"]["by_type"].items():
                agg_by_type[k] = agg_by_type.get(k, 0) + v
            agg_total += sa["tool_calls"]["total"]
            agg_errors += sa["tool_calls"]["errors"]

        # ── Timing ────────────────────────────────────────────────────────
        subagent_active_s = sum(
            sa["timing"]["agent_active_s"] or 0 for sa in subagent_details.values()
        )
        main_active_s = main["timing"]["agent_active_s"] or 0

        # ── Copilot version ───────────────────────────────────────────────
        copilot_version = None
        for e in main_events:
            if e.get("type") == "session_start":
                copilot_version = e.get("attrs", {}).get("copilotVersion")
                break

        # ── Assemble output ───────────────────────────────────────────────
        out: dict[str, Any] = {
            "_meta": {
                "session_id": main["session_id"] or src.session_id,
                # sessionTitle is available in agent-debug-log.json (not parsed here)
                "session_title": None,
                "agent_name": main["agent_name"],
                "model": main["model"],
                "reasoning_effort": main["reasoning_effort"],
                "copilot_version": copilot_version,
                "is_baseline": False,
            },
            "context": {
                "loaded_agents": main["discovery"]["agents"],
                "loaded_skills": main["discovery"]["skills"],
                "loaded_instructions": main["discovery"]["instructions"],
            },
            "interaction": {
                "user_turn_count": len(main["user_messages"]),
                "user_messages": main["user_messages"],
                "initial_prompt": main["user_messages"][0]
                if main["user_messages"]
                else None,
                "final_answer": (
                    main["final_responses"][-1]["text"]
                    if main["final_responses"]
                    else None
                ),
                "all_final_responses": main["final_responses"],
            },
            "timing": {
                "session_start_ms": main["timing"]["session_start_ms"],
                "session_end_ms": main["timing"]["session_end_ms"],
                # Spans entire human session including user think-time between turns
                "session_wall_time_s": main["timing"]["wall_time_s"],
                # Sum of turn durations = actual agent processing time
                "main_agent_active_s": main_active_s,
                "subagents_active_s": round(subagent_active_s, 3),
                "total_agent_active_s": round(main_active_s + subagent_active_s, 3),
                "per_turn": [
                    {"turn_id": t["turn_id"], "duration_s": t["duration_s"]}
                    for t in main["agent_turns"]
                ],
            },
            "tokens": {
                "main_only": {
                    "input": main["tokens"]["input_total"],
                    "output": main["tokens"]["output_total"],
                    "cached": main["tokens"]["cached_total"],
                    "llm_calls": main["tokens"]["llm_calls"],
                },
                "all_agents": {
                    "input": total_input,
                    "output": total_output,
                    "cached": total_cached,
                    "llm_calls": total_llm_calls,
                    "cache_hit_rate": (
                        round(total_cached / total_input, 4) if total_input else 0
                    ),
                },
                "per_llm_call": main["tokens"]["per_call"],
            },
            "tool_calls": {
                "main_agent": {
                    "total": main["tool_calls"]["total"],
                    "errors": main["tool_calls"]["errors"],
                    "by_type": main["tool_calls"]["by_type"],
                    "duration_s_by_type": main["tool_calls"]["duration_by_type"],
                    "error_list": main["tool_calls"]["error_list"],
                    "terminal_commands": main["tool_calls"]["terminal_commands"],
                },
                "all_agents": {
                    "total": agg_total,
                    "errors": agg_errors,
                    "by_type": agg_by_type,
                },
            },
            "artifacts": main["artifacts"],
            "subagents": [],
            "main_turns": main["agent_turns"],
        }

        # ── Artifact reconstruction ───────────────────────────────────────
        # Collect all create_file / apply_patch events from main + subagents.
        # main_events is already in memory; subagent events are in _sa_raw_events.
        _write_events = [
            e
            for e in main_events
            if e.get("type") == "tool_call"
            and e.get("name") in ("create_file", "apply_patch")
        ]
        for _sa_events in _sa_raw_events.values():
            _write_events.extend(
                e
                for e in _sa_events
                if e.get("type") == "tool_call"
                and e.get("name") in ("create_file", "apply_patch")
            )
        out["artifact_contents"] = reconstruct_artifacts(_write_events)

        # ── Subagent summaries ────────────────────────────────────────────
        for ref in main["subagent_calls"]:
            fname = ref.get("child_file", "")
            detail = subagent_details.get(fname)
            out["subagents"].append(
                {
                    "agent_name": ref.get("agent_name"),
                    "call_id": ref.get("call_id"),
                    "child_file": fname,
                    "parent_turn_id": ref.get("turn_id"),
                    "dur_s": ref.get("dur_s"),
                    "structured_output": ref.get("structured_output"),
                    "user_prompt": (
                        detail["user_messages"][0]
                        if detail and detail["user_messages"]
                        else None
                    ),
                    "final_response": (
                        detail["final_responses"][-1]["text"]
                        if detail and detail["final_responses"]
                        else None
                    ),
                    "tokens": detail["tokens"] if detail else None,
                    "tool_calls": {
                        "total": detail["tool_calls"]["total"],
                        "errors": detail["tool_calls"]["errors"],
                        "by_type": detail["tool_calls"]["by_type"],
                        "duration_s_by_type": detail["tool_calls"]["duration_by_type"],
                        "error_list": detail["tool_calls"]["error_list"],
                        "terminal_commands": detail["tool_calls"]["terminal_commands"],
                    }
                    if detail
                    else None,
                    "timing": detail["timing"] if detail else None,
                    "artifacts": detail["artifacts"] if detail else [],
                    "turns": detail["agent_turns"] if detail else [],
                }
            )

    return out
