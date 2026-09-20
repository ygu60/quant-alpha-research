"""Live terminal dashboard for the quant research pipeline.

Tails logs/events.jsonl (written by run_pipeline.py / src/events.py) and
shows: recent status/action events, the latest PnL + metrics per strategy,
open bugs, and the health of the robinhood-trading MCP connection.

This reads local files and calls `claude mcp list` as a read-only health
check -- it never places trades or touches account data.

Run: python dashboard.py   (Ctrl+C to quit)
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import threading
import time

# Windows consoles often default to a legacy codepage (cp1252) that can't
# encode the box-drawing/braille characters rich and plotext emit; force
# UTF-8 so the dashboard doesn't crash on those terminals.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src import events

REFRESH_SECS = 1.0
MCP_CHECK_INTERVAL_SECS = 30.0
MAX_EVENTS_SHOWN = 14

KIND_STYLE = {
    "action": "cyan",
    "status": "green",
    "warning": "yellow",
    "bug": "bold red",
    "info": "dim",
}

METRIC_COLUMNS = ["sharpe", "sortino", "annualized_return", "annualized_vol", "max_drawdown", "hit_rate"]

CLAUDE_BIN = shutil.which("claude") or r"C:\Users\Owner\.local\bin\claude.exe"
PROJECT_ROOT = Path(__file__).resolve().parent


class McpStatus:
    """Background poller for `claude mcp get robinhood-trading` (read-only)."""

    def __init__(self) -> None:
        self.text = "checking..."
        self.style = "dim"
        self.checked_at: str | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def snapshot(self) -> tuple[str, str, str | None]:
        with self._lock:
            return self.text, self.style, self.checked_at

    def _check_once(self) -> None:
        try:
            result = subprocess.run(
                [CLAUDE_BIN, "mcp", "get", "robinhood-trading"],
                capture_output=True, text=True, timeout=15, cwd=PROJECT_ROOT,
            )
            out = result.stdout.strip()
            if "Connected" in out or "\u221a" in out:
                text, style = "Connected", "bold green"
            elif "Needs authentication" in out:
                text, style = "Needs authentication", "bold yellow"
            elif result.returncode != 0:
                text, style = "claude CLI error (see logs)", "bold red"
                events.log("warning", "claude mcp health check failed", stderr=result.stderr.strip()[:500])
            else:
                text, style = "Unknown", "dim"
        except FileNotFoundError:
            text, style = "claude CLI not found on PATH", "bold red"
        except subprocess.TimeoutExpired:
            text, style = "health check timed out", "yellow"
        with self._lock:
            self.text = text
            self.style = style
            self.checked_at = datetime.now().strftime("%H:%M:%S")

    def run_forever(self) -> None:
        while not self._stop.is_set():
            self._check_once()
            self._stop.wait(MCP_CHECK_INTERVAL_SECS)

    def stop(self) -> None:
        self._stop.set()


def render_events_panel() -> Panel:
    records = events.read_all()[-MAX_EVENTS_SHOWN:]
    table = Table(expand=True, show_edge=False, pad_edge=False)
    table.add_column("Time", style="dim", width=8)
    table.add_column("Kind", width=8)
    table.add_column("Message", ratio=1)
    for rec in reversed(records):
        ts = rec.get("ts", "")
        try:
            clock = datetime.fromisoformat(ts).astimezone().strftime("%H:%M:%S")
        except ValueError:
            clock = ts[:8]
        kind = rec.get("kind", "info")
        style = KIND_STYLE.get(kind, "white")
        table.add_row(clock, Text(kind, style=style), rec.get("message", ""))
    if not records:
        table.add_row("", "", "no events yet -- run: python run_pipeline.py")
    return Panel(table, title="Recent Activity", border_style="blue")


def render_strategy_panel() -> Panel:
    records = events.read_all()
    latest: dict[str, dict] = {}
    for rec in records:
        if rec.get("kind") == "status":
            data = rec.get("data", {})
            name = data.get("strategy")
            if name:
                latest[name] = data

    table = Table(expand=True, show_edge=False, pad_edge=False)
    table.add_column("Strategy", ratio=2)
    table.add_column("PnL", justify="right")
    for col in METRIC_COLUMNS:
        table.add_column(col, justify="right")

    for name, data in latest.items():
        curve = data.get("equity_curve")
        if curve and curve.get("values"):
            pnl_pct = (curve["values"][-1] - 1.0) * 100
            pnl_cell = Text(f"{pnl_pct:+.2f}%", style="green" if pnl_pct >= 0 else "red")
        else:
            pnl_cell = Text("-")
        row = [name, pnl_cell]
        for col in METRIC_COLUMNS:
            val = data.get(col)
            row.append(f"{val:.3f}" if isinstance(val, (int, float)) else "-")
        table.add_row(*row)

    if not latest:
        table.add_row("no strategy runs logged yet", "-", *(["-"] * len(METRIC_COLUMNS)))
    return Panel(table, title="Latest Strategy Metrics (from logged runs)", border_style="green")


ORDER_STATE_STYLE = {
    "queued": "yellow",
    "unconfirmed": "yellow",
    "confirmed": "yellow",
    "pending": "yellow",
    "partially_filled": "bold yellow",
    "filled": "green",
    "cancelled": "dim",
    "canceled": "dim",
    "rejected": "bold red",
    "failed": "bold red",
}


def render_orderflow_panel() -> Panel:
    records = [r for r in events.read_all() if r.get("kind") == "order"]
    latest_by_id: dict[str, dict] = {}
    for rec in records:
        data = rec.get("data", {})
        order_id = data.get("order_id")
        if order_id:
            latest_by_id[order_id] = rec  # last write wins -> most recent state

    orders = sorted(latest_by_id.values(), key=lambda r: r.get("ts", ""), reverse=True)
    active = [r for r in orders if r.get("data", {}).get("state") in events.ACTIVE_ORDER_STATES]

    table = Table(expand=True, show_edge=False, pad_edge=False)
    table.add_column("Time", style="dim", width=8)
    table.add_column("Symbol", width=8)
    table.add_column("Side", width=5)
    table.add_column("Type", width=7)
    table.add_column("Qty / $", justify="right")
    table.add_column("State")

    shown = active if active else orders[:MAX_EVENTS_SHOWN]
    for rec in shown:
        data = rec.get("data", {})
        ts = rec.get("ts", "")
        try:
            clock = datetime.fromisoformat(ts).astimezone().strftime("%H:%M:%S")
        except ValueError:
            clock = ts[:8]
        state = data.get("state", "-")
        size = data.get("dollar_amount")
        size_str = f"${size}" if size is not None else str(data.get("quantity", "-"))
        table.add_row(
            clock,
            data.get("symbol", "-"),
            data.get("side", "-"),
            data.get("order_type", "-"),
            size_str,
            Text(state, style=ORDER_STATE_STYLE.get(state, "white")),
        )

    if not orders:
        table.add_row("", "", "", "", "", "no orders logged yet")

    title = f"Order Flow ({len(active)} active)" if orders else "Order Flow"
    return Panel(table, title=title, border_style="cyan")


def render_bugs_panel() -> Panel:
    bugs = [r for r in events.read_all() if r.get("kind") == "bug"]
    if not bugs:
        body = Text("No bugs logged.", style="dim")
    else:
        last = bugs[-1]
        body = Text()
        body.append(f"{len(bugs)} bug(s) logged. Most recent:\n", style="bold red")
        body.append(f"  {last.get('message', '')}\n")
        err = last.get("data", {}).get("error")
        if err:
            body.append(f"  {err}\n", style="red")
    return Panel(body, title="Bugs", border_style="red")


def render_mcp_panel(mcp: McpStatus) -> Panel:
    text, style, checked_at = mcp.snapshot()
    body = Text()
    body.append("robinhood-trading: ", style="bold")
    body.append(f"{text}\n", style=style)
    body.append(f"last checked: {checked_at or 'pending'}\n", style="dim")
    body.append("(read-only connection health check -- no trades placed by this dashboard)", style="dim italic")
    return Panel(body, title="Robinhood MCP Connection", border_style="magenta")


def build_layout(mcp: McpStatus) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3),
    )
    layout["body"].split_row(
        Layout(name="left"),
        Layout(name="right"),
    )
    layout["left"].split_column(
        Layout(name="events"),
        Layout(name="orderflow", size=10),
        Layout(name="bugs", size=8),
    )
    layout["right"].split_column(
        Layout(name="strategies"),
        Layout(name="mcp", size=7),
    )

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    layout["header"].update(
        Panel(Text(f"Quant Alpha Research -- Live Dashboard    {now}", justify="center", style="bold white"),
              style="on grey15")
    )
    layout["events"].update(render_events_panel())
    layout["orderflow"].update(render_orderflow_panel())
    layout["bugs"].update(render_bugs_panel())
    layout["strategies"].update(render_strategy_panel())
    layout["mcp"].update(render_mcp_panel(mcp))
    layout["footer"].update(
        Panel(Text(f"log file: {events.LOG_FILE}    |    Ctrl+C to quit", justify="center", style="dim"))
    )
    return layout


def main() -> None:
    console = Console()
    mcp = McpStatus()
    poller = threading.Thread(target=mcp.run_forever, daemon=True)
    poller.start()

    try:
        with Live(build_layout(mcp), console=console, refresh_per_second=1, screen=True) as live:
            while True:
                time.sleep(REFRESH_SECS)
                live.update(build_layout(mcp))
    except KeyboardInterrupt:
        pass
    finally:
        mcp.stop()


if __name__ == "__main__":
    main()
