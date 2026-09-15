"""
session-parser — GitHub Copilot agent session analysis library.

Public API::

    from session_parser import parse_session, parse_agent_jsonl, SessionSource

    result = parse_session("session.zip")
    result = parse_session("extracted/a6ca3ad3-.../")
"""

from .helpers import classify_terminal_command, parse_terminal_commands
from .io import SessionSource, iter_jsonl
from .parser import parse_agent_jsonl, parse_session
from .writers import write_output_dir

__all__ = [
    "SessionSource",
    "classify_terminal_command",
    "iter_jsonl",
    "parse_agent_jsonl",
    "parse_session",
    "parse_terminal_commands",
    "write_output_dir",
]
