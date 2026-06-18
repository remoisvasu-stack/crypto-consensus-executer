"""binance_exec.py — Binance European Options order placement (dry / live).

NET-NEW for Mark-2. The collector/brain only ever touch *public* market data; this
is the first component that authenticates and can move real money. It is written
dry-run-first: with LIVE=false it logs the exact signed request it WOULD send and
returns a simulated fill, so the whole pipeline can be validated end-to-end before
a single real order.

Binance options REST: POST /eapi/v1/order (SIGNED). Symbol format is
    BTC-<YYMMDD>-<STRIKE>-<C|P>     e.g. BTC-260626-66000-C

Before sending, it re-prices the order against a FRESH public mark
(GET /eapi/v1/mark) fetched at order time, instead of trusting the brain's
~1-min-old forwarded premium. This also doubles as a live-market validation:
no fresh quote (bad symbol / illiquid / geo-blocked host) -> fall back to the
forwarded price. Disable with REPRICE_AT_ORDER=false.

NOTE: order quantity/price precision and the exact contract multiplier vary by
symbol — validate against /eapi/v1/exchangeInfo before going live. This module
keeps the request shape correct and the auth real; the tuning is intentionally
left explicit rather than hidden.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
import time
from typing import Optional, Literal
from urllib.parse import urlencode

import httpx

log = logging.getLogger("executor.binance")

EAPI = os.getenv("BINANCE_EAPI") or "https://eapi.binance.com"
API_KEY = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")
RECV_WINDOW = int(os.getenv("BINANCE_RECV_WINDOW", "5000"))
REPRICE = os.getenv("REPRICE_AT_ORDER", "true").lower() == "true"


def mark_price(symbol: str, timeout: float = 8.0) -> Optional[float]:
    """Current Binance option mark price for `symbol` (public /eapi/v1/mark, no auth).

    Lets the executor price/validate an order against the live market at send
    time rather than the brain's stale forwarded premium. Returns None if the
    quote is unavailable (bad symbol, illiquid, or host geo-blocked), so the
    caller can fall back to the forwarded price.
    """
    try:
        r = httpx.get(f"{EAPI}/eapi/v1/mark", params={"symbol": symbol}, timeout=timeout)
        if r.status_code >= 300:
            log.warning("mark %s -> %s %s", symbol, r.status_code, r.text[:200])
            return None
        data = r.json()
        rec = (data[0] if isinstance(data, list) and data else data) or {}
        mp = float(rec.get("markPrice", 0) or 0)
        return mp if mp > 0 else None
    except Exception as e:
        log.warning("mark fetch %s failed: %s", symbol, e)
        return None


def option_symbol(expiry: str, strike: float, side: str) -> str:
    """Build a Binance option symbol. `expiry` is the YYMMDD label from the feed."""
    cp = "C" if side.upper() == "CALL" else "P"
    return f"BTC-{expiry}-{int(round(strike))}-{cp}"


def _sign(params: dict) -> str:
    query = urlencode(params)
    sig = hmac.new(API_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
    return f"{query}&signature={sig}"


def credentials_ok() -> bool:
    return bool(API_KEY and API_SECRET)


def place_order(*, symbol: str, action: Literal["BUY", "SELL"], qty: float,
                price: Optional[float], live: bool,
                tif: str = "GTC", reprice: bool = REPRICE) -> dict:
    """Place one option order. `action` BUY opens a long, SELL closes it.

    When `reprice` (default REPRICE_AT_ORDER env), the LIMIT price is taken from a
    FRESH Binance mark fetched now; if that's unavailable it falls back to the
    brain's forwarded `price`. Both are returned (fresh_mark / forwarded_price)
    for transparency. Returns {ok, dry, symbol, ...} — never raises; failures come
    back as ok=False so the caller can record and continue.
    """
    fresh = mark_price(symbol) if reprice else None
    limit = fresh or price                       # fresh mark wins; else forwarded
    params = {
        "symbol": symbol,
        "side": action,
        "type": "LIMIT" if limit else "MARKET",
        "quantity": qty,
        "timestamp": int(time.time() * 1000),
        "recvWindow": RECV_WINDOW,
    }
    if limit:
        params["price"] = limit
        params["timeInForce"] = tif

    base = {"symbol": symbol, "action": action, "qty": qty, "price": limit,
            "fresh_mark": fresh, "forwarded_price": price}

    if not live:
        log.info("DRY-RUN: would POST %s/eapi/v1/order %s (fresh_mark=%s forwarded=%s)",
                 EAPI, params, fresh, price)
        return {"ok": True, "dry": True, **base, "request": params}

    if not credentials_ok():
        return {"ok": False, "dry": False, **base, "error": "missing_api_credentials"}

    try:
        body = _sign(params)
        r = httpx.post(f"{EAPI}/eapi/v1/order", content=body,
                       headers={"X-MBX-APIKEY": API_KEY,
                                "Content-Type": "application/x-www-form-urlencoded"},
                       timeout=15.0)
        ok = r.status_code < 300
        if not ok:
            log.warning("Order %s %s -> %s %s", action, symbol,
                        r.status_code, r.text[:300])
        return {"ok": ok, "dry": False, **base,
                "status": r.status_code, "response": r.json() if ok else r.text[:300]}
    except Exception as e:
        log.exception("Order placement failed")
        return {"ok": False, "dry": False, **base, "error": str(e)}
