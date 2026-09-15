"""
io.py — I/O abstraction for session source files.

Provides :class:`SessionSource`, which presents a uniform interface over two
physical layouts:

* A ``.zip`` file produced by the GitHub Copilot debug-log export.
* A pre-extracted directory with the same internal structure.

Both layouts share the same inner file names:
``main.jsonl``, ``runSubagent-*.jsonl``, ``title-*.jsonl``,
``system_prompt_0.json``, ``tools_0.json``, ``models.json``.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any, Iterator


def iter_jsonl(source) -> Iterator[dict]:
    """
    Yield parsed JSON objects from a JSONL source.

    *source* may be:
    * A :class:`pathlib.Path` or ``str`` path to a file on disk.
    * Any file-like object (text or binary with UTF-8 encoding).

    Malformed lines are silently skipped.
    """
    if isinstance(source, (str, Path)):
        with open(source, encoding='utf-8', errors='replace') as fh:
            yield from iter_jsonl(fh)
        return
    for raw in source:
        line = raw.strip() if isinstance(raw, str) else raw.strip()
        if line:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                pass


class SessionSource:
    """
    Uniform read-only access to a Copilot debug-log session.

    Accepts either a ``.zip`` file or a pre-extracted directory.
    The *session_id* attribute is inferred from the zip's inner directory name
    or the directory name itself.

    Usage::

        src = SessionSource("session.zip")
        events = list(src.iter_jsonl("main.jsonl"))
        src.close()

    Or as a context manager::

        with SessionSource("session/") as src:
            for fname in src.list_jsonl():
                ...
    """

    def __init__(self, path: str | Path) -> None:
        path = Path(path)

        if path.is_file() and path.suffix == '.zip':
            self._zip = zipfile.ZipFile(path, 'r')
            names = self._zip.namelist()
            # All members live under a single top-level directory named after the session UUID.
            self._prefix = names[0].split('/')[0] + '/' if names else ''
            self._members: dict[str, Any] = {
                n[len(self._prefix):]: n
                for n in names
                if n != self._prefix and not n.endswith('/')
            }
            self._dir: Path | None = None
            self.session_id: str = self._prefix.rstrip('/')

        elif path.is_dir():
            self._zip = None
            self._dir = path
            self.session_id = path.name
            self._members = {f.name: f for f in path.iterdir() if f.is_file()}

        else:
            raise ValueError(
                f'Expected a .zip file or an extracted session directory, got: {path}'
            )

    # ── Public interface ──────────────────────────────────────────────────

    def list_jsonl(self) -> list[str]:
        """Return names of all ``.jsonl`` members in the session."""
        return [n for n in self._members if n.endswith('.jsonl')]

    def iter_jsonl(self, name: str) -> Iterator[dict]:
        """Yield parsed events from a named ``.jsonl`` member."""
        if name not in self._members:
            return
        if self._zip:
            with self._zip.open(self._members[name]) as raw:
                reader = io.TextIOWrapper(raw, encoding='utf-8', errors='replace')
                yield from iter_jsonl(reader)
        else:
            yield from iter_jsonl(self._members[name])

    def read_json(self, name: str) -> Any:
        """Parse and return a named JSON member, or ``None`` if absent."""
        if name not in self._members:
            return None
        if self._zip:
            with self._zip.open(self._members[name]) as fh:
                return json.load(fh)
        else:
            with open(self._members[name], encoding='utf-8') as fh:
                return json.load(fh)

    def close(self) -> None:
        """Release the underlying zip handle if open."""
        if self._zip:
            self._zip.close()

    # ── Context manager ───────────────────────────────────────────────────

    def __enter__(self) -> 'SessionSource':
        return self

    def __exit__(self, *_) -> None:
        self.close()
