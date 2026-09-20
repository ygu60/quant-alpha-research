"""Lightweight structured event log shared by the pipeline and the dashboard.

Events are appended as JSON lines to logs/events.jsonl so the dashboard
(dashboard.py) can tail them without importing pipeline code or sharing
process memory. Kinds: "action" (something started), "status" (something
finished, usually with a `data` payload of metrics), "warning", "bug"
(caught exception), "info".
"""
from __future__ import annotations

import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_FILE = LOG_DIR / "events.jsonl"

_VALID_KINDS = {"action", "status", "warning", "bug", "info", "order"}

# Order states considered still "live" -- everything else (filled, cancelled,
# rejected, failed, etc.) is treated as terminal/history in the dashboard.
ACTIVE_ORDER_STATES = {"queued", "unconfirmed", "confirmed", "partially_filled", "pending"}


def log(kind: str, message: str, **data: Any) -> None:
    if kind not in _VALID_KINDS:
        raise ValueError(f"unknown event kind: {kind!r} (expected one of {_VALID_KINDS})")
    LOG_DIR.mkdir(exist_ok=True)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kind": kind,
        "message": message,
    }
    if data:
        record["data"] = data
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def log_order(order_id: str, symbol: str, side: str, order_type: str, state: str, **data: Any) -> None:
    """Log an order placement or a status update for an existing order.

    The dashboard's Order Flow panel keeps only the most recent event per
    order_id, so re-logging the same order_id with a new `state` (e.g. after
    polling get_equity_orders) updates its row instead of duplicating it.
    """
    log(
        "order",
        f"{side} {symbol} ({order_type}) -> {state}",
        order_id=order_id, symbol=symbol, side=side, order_type=order_type, state=state,
        **data,
    )


def log_bug(message: str, exc: BaseException) -> None:
    log(
        "bug",
        message,
        error=f"{type(exc).__name__}: {exc}",
        traceback=traceback.format_exc(),
    )


def read_all() -> list[dict]:
    if not LOG_FILE.exists():
        return []
    records = []
    with LOG_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records
