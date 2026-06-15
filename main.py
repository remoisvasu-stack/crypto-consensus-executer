"""main.py — Executor service (Render Frankfurt, or any non-US region).

Part 3 of 3 (see ../README.md). Two roles, both needing Binance reachability:
  1. LIVE FEED — polls the Binance option chain and forwards {row, chain} to the
     Brain (the HF Space is geo-blocked from Binance, so the Brain can't self-feed).
     This lets the collector stay training-only. Off unless BRAIN_URL is set.
  2. EXECUTION — receives fired order intents from the Brain and mirrors them as
     real Binance option orders — each `opened` is a BUY, each `closed` a SELL.

The brain's paper trader is the single source of truth, so live should track paper
minus slippage/fees.

Safety first:
  * Starts in DRY-RUN (LIVE defaults false). Flip live<->dry at runtime via
    POST /admin/mode — no redeploy.
  * POST /admin/halt = kill switch (blocks all new orders).
  * Idempotent: every open/close is de-duplicated by (strategy, ts).
  * Per-strategy + global open caps; daily realized-loss kill switch.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Request

import binance_exec as bx
import webhook
from core.binance_feed import BinanceOptionsFeed

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s | %(message)s")
log = logging.getLogger("executor")

# --- config ---------------------------------------------------------------
EXECUTOR_TOKEN = os.getenv("EXECUTOR_TOKEN", "")
MAX_OPEN_PER_STRATEGY = int(os.getenv("MAX_OPEN_PER_STRATEGY", "1"))
MAX_OPEN_TOTAL = int(os.getenv("MAX_OPEN_TOTAL", "12"))
DAILY_LOSS_LIMIT = float(os.getenv("DAILY_LOSS_LIMIT", "0"))   # 0 = disabled (USDT)
DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
STATE_PATH = DATA_DIR / "executor_state.json"

# --- live feed -> brain (this host reaches Binance; the HF brain does not) ------
# The executor also polls Binance and forwards {row, chain} to the brain, so the
# collector can stay training-only. Forwarding is off unless BRAIN_URL is set.
BRAIN_URL = os.getenv("BRAIN_URL", "").rstrip("/")
WEBHOOK_TOKEN = os.getenv("WEBHOOK_TOKEN", "")
POLL_INTERVAL = float(os.getenv("POLL_INTERVAL", "60"))
SKIP_SAME_DAY = os.getenv("SKIP_SAME_DAY_EXPIRY", "true").lower() == "true"
MIN_EXPIRY_HOURS = float(os.getenv("MIN_EXPIRY_HOURS", "3"))

# --- state ----------------------------------------------------------------
_lock = threading.Lock()
state = {
    "live": os.getenv("LIVE", "false").lower() == "true",
    "halted": False,
    "open": {},          # strategy -> {symbol, qty, entry_prem, entry_ts}
    "seen": [],          # de-dup keys "open:strategy:ts" / "close:strategy:ts"
    "realized_pnl_today": 0.0,
    "pnl_day": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    "orders": 0, "fills": 0, "rejected": 0,
}
feed_stats = {"ticks": 0, "forwarded": 0, "last_tick_ts": None, "last_error": None}


def _load_state() -> None:
    if STATE_PATH.exists():
        try:
            with _lock:
                state.update(json.loads(STATE_PATH.read_text()))
        except Exception:
            log.exception("Failed to load executor state")


def _save_state() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        STATE_PATH.write_text(json.dumps(state, indent=2, default=str))
    except Exception:
        log.exception("Failed to save executor state")


def _roll_day() -> None:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if state["pnl_day"] != today:
        state["pnl_day"] = today
        state["realized_pnl_today"] = 0.0


def _seen(key: str) -> bool:
    if key in state["seen"]:
        return True
    state["seen"].append(key)
    state["seen"] = state["seen"][-5000:]   # bound memory
    return False


def _handle_open(o: dict, expiry: str) -> dict:
    strat = o["strategy"]
    key = f"open:{strat}:{o.get('entry_ts')}"
    if _seen(key):
        return {"strategy": strat, "skipped": "duplicate"}
    if sum(1 for s in state["open"] if s == strat) >= MAX_OPEN_PER_STRATEGY:
        return {"strategy": strat, "skipped": "per_strategy_cap"}
    if len(state["open"]) >= MAX_OPEN_TOTAL:
        return {"strategy": strat, "skipped": "global_cap"}

    symbol = bx.option_symbol(expiry, o["strike"], o["side"])
    res = bx.place_order(symbol=symbol, action="BUY", qty=o["qty"],
                         price=o.get("entry_prem"), live=state["live"])
    state["orders"] += 1
    if res.get("ok"):
        state["fills"] += 1
        state["open"][strat] = {"symbol": symbol, "qty": o["qty"],
                                "entry_prem": o.get("entry_prem"),
                                "entry_ts": o.get("entry_ts")}
    else:
        state["rejected"] += 1
    return {"strategy": strat, "symbol": symbol, "order": res}


def _handle_close(c: dict, expiry: str) -> dict:
    strat = c["strategy"]
    key = f"close:{strat}:{c.get('exit_ts')}"
    if _seen(key):
        return {"strategy": strat, "skipped": "duplicate"}
    pos = state["open"].get(strat)
    symbol = (pos or {}).get("symbol") or bx.option_symbol(expiry, c["strike"], c["side"])

    res = bx.place_order(symbol=symbol, action="SELL", qty=c["qty"],
                         price=c.get("exit_prem"), live=state["live"])
    state["orders"] += 1
    if res.get("ok"):
        state["fills"] += 1
        entry = (pos or {}).get("entry_prem") or 0.0
        realized = (c.get("exit_prem", 0.0) - entry) * c["qty"]
        state["realized_pnl_today"] += realized
        state["open"].pop(strat, None)
        # Daily-loss kill switch.
        if DAILY_LOSS_LIMIT > 0 and state["realized_pnl_today"] <= -DAILY_LOSS_LIMIT:
            state["halted"] = True
            log.warning("DAILY LOSS LIMIT hit (%.2f) — HALTED",
                        state["realized_pnl_today"])
    else:
        state["rejected"] += 1
    return {"strategy": strat, "symbol": symbol, "order": res}


def _on_row(row: dict, chain: list) -> None:
    """Feed-loop callback: forward each {row, chain} to the brain (fire-and-forget)."""
    feed_stats["ticks"] += 1
    feed_stats["last_tick_ts"] = row.get("ts")
    if not BRAIN_URL:
        return
    if webhook.post_json(f"{BRAIN_URL}/tick", {"row": row, "chain": chain},
                         token=WEBHOOK_TOKEN):
        feed_stats["forwarded"] += 1
    else:
        feed_stats["last_error"] = "forward_failed"


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_state()
    task = None
    if BRAIN_URL:
        feed = BinanceOptionsFeed(skip_same_day=SKIP_SAME_DAY,
                                  min_expiry_hours=MIN_EXPIRY_HOURS)
        task = asyncio.create_task(feed.run_loop(_on_row, interval=POLL_INTERVAL))
        log.info("Live feed -> brain started (poll %.0fs -> %s)", POLL_INTERVAL, BRAIN_URL)
    else:
        log.info("BRAIN_URL unset — live feed disabled (order execution only)")
    yield
    if task:
        task.cancel()


app = FastAPI(title="BTC Executor", lifespan=lifespan)


@app.get("/health")
def health():
    with _lock:
        return {
            "ok": True, "service": "executor",
            "mode": "LIVE" if state["live"] else "DRY-RUN",
            "halted": state["halted"],
            "open_positions": len(state["open"]),
            "credentials": bx.credentials_ok(),
            "realized_pnl_today": round(state["realized_pnl_today"], 4),
            "orders": state["orders"], "fills": state["fills"],
            "rejected": state["rejected"],
            "feed": bool(BRAIN_URL),
            "ticks": feed_stats["ticks"], "forwarded": feed_stats["forwarded"],
            "last_tick_ts": feed_stats["last_tick_ts"],
        }


@app.post("/signal")
async def signal(request: Request,
                 x_executor_token: Optional[str] = Header(default=None)):
    if EXECUTOR_TOKEN and x_executor_token != EXECUTOR_TOKEN:
        raise HTTPException(401, "bad token")
    payload = await request.json()
    expiry = payload.get("expiry")
    results = {"opened": [], "closed": []}
    with _lock:
        _roll_day()
        if state["halted"]:
            return {"halted": True, "note": "executor halted — no orders placed"}
        # Close first (free up caps), then open.
        for c in payload.get("closed", []):
            results["closed"].append(_handle_close(c, expiry))
        for o in payload.get("opened", []):
            results["opened"].append(_handle_open(o, expiry))
        _save_state()
    return {"mode": "LIVE" if state["live"] else "DRY-RUN", **results}


@app.post("/admin/mode")
async def set_mode(request: Request,
                   x_executor_token: Optional[str] = Header(default=None)):
    """Flip live<->dry at runtime. Body: {"live": true|false}."""
    if EXECUTOR_TOKEN and x_executor_token != EXECUTOR_TOKEN:
        raise HTTPException(401, "bad token")
    body = await request.json()
    with _lock:
        state["live"] = bool(body.get("live"))
        _save_state()
    log.warning("MODE set to %s", "LIVE" if state["live"] else "DRY-RUN")
    return {"mode": "LIVE" if state["live"] else "DRY-RUN"}


@app.post("/admin/halt")
def halt(x_executor_token: Optional[str] = Header(default=None), resume: bool = False):
    """Kill switch. `?resume=true` clears it."""
    if EXECUTOR_TOKEN and x_executor_token != EXECUTOR_TOKEN:
        raise HTTPException(401, "bad token")
    with _lock:
        state["halted"] = not resume
        _save_state()
    return {"halted": state["halted"]}
