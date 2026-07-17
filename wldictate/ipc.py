"""JSON-lines IPC between the tray app and the dictation worker.

Commands (tray -> worker stdin), one JSON object per line:
    {"cmd": "start", "device": 3}                 device optional
    {"cmd": "start", "mode": "contextual"}        mode optional (default standard)
    {"cmd": "stop"}
    {"cmd": "quit"}

Events (worker -> tray stdout), one JSON object per line:
    {"ev": "ready"}                       model loaded and warmed
    {"ev": "listening"}                   audio session active
    {"ev": "stopped"}                     audio session ended
    {"ev": "commit", "text": "..."}       text committed (already typed)
    {"ev": "error", "msg": "..."}         recoverable error
    {"ev": "log", "msg": "..."}           informational

Any non-JSON line on either channel is treated as plain log text: third-party
libraries occasionally print to stdout, and that must never break the protocol.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

VALID_COMMANDS = ("start", "stop", "quit")
VALID_EVENTS = ("ready", "listening", "stopped", "commit", "error", "log")
VALID_MODES = ("standard", "contextual")


@dataclass(frozen=True)
class Command:
    cmd: str
    device: int | None = None
    # Authoritative device identity: Pulse/PipeWire indices drift as streams
    # appear/disappear, so the worker re-resolves by name at start time.
    device_name: str | None = None
    # Dictation mode for "start"; unknown/absent values parse as "standard"
    # (old workers ignore the key entirely — forward/backward compatible).
    mode: str = "standard"


@dataclass(frozen=True)
class Event:
    ev: str
    text: str | None = None
    msg: str | None = None


def format_command(
    cmd: str,
    device: int | None = None,
    device_name: str | None = None,
    mode: str | None = None,
) -> str:
    payload: dict[str, Any] = {"cmd": cmd}
    if device is not None:
        payload["device"] = device
    if device_name is not None:
        payload["device_name"] = device_name
    if mode is not None:
        payload["mode"] = mode
    return json.dumps(payload)


def parse_command(line: str) -> Command | None:
    """Parse a command line; returns None for junk/unknown input."""
    obj = _parse_obj(line)
    if obj is None:
        return None
    cmd = obj.get("cmd")
    if cmd not in VALID_COMMANDS:
        return None
    device = obj.get("device")
    if device is not None and not isinstance(device, int):
        device = None
    device_name = obj.get("device_name")
    if device_name is not None and not isinstance(device_name, str):
        device_name = None
    mode = obj.get("mode")
    if mode not in VALID_MODES:
        mode = "standard"
    return Command(cmd=cmd, device=device, device_name=device_name, mode=mode)


def format_event(ev: str, *, text: str | None = None, msg: str | None = None) -> str:
    payload: dict[str, Any] = {"ev": ev}
    if text is not None:
        payload["text"] = text
    if msg is not None:
        payload["msg"] = msg
    return json.dumps(payload)


def parse_event(line: str) -> Event | None:
    """Parse an event line; non-JSON/unknown lines return None (treat as log)."""
    obj = _parse_obj(line)
    if obj is None:
        return None
    ev = obj.get("ev")
    if ev not in VALID_EVENTS:
        return None
    text = obj.get("text")
    msg = obj.get("msg")
    return Event(
        ev=ev,
        text=text if isinstance(text, str) else None,
        msg=msg if isinstance(msg, str) else None,
    )


def _parse_obj(line: str) -> dict[str, Any] | None:
    line = line.strip()
    if not line.startswith("{"):
        return None
    try:
        obj = json.loads(line)
    except (json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None
