"""core/binance_feed.py — Live BTC option-chain collector for Binance Options.

Replaces the entire Upstox + Google Sheets + Apps Script ingest layer. Polls the
public Binance Options REST API (no auth needed for market data), merges the
per-endpoint payloads into a single per-contract list, and hands it to
`core.features.build_feature_row` to produce one model-ready row per cycle.

Endpoints (eapi.binance.com unless noted):
    /eapi/v1/exchangeInfo            symbol → side/strike/expiry meta   (cached)
    /eapi/v1/index?underlying=…      BTC index price (spot)
    /eapi/v1/mark                    markIV + Greeks, ALL symbols (1 call)
    /eapi/v1/openInterest?…          OI per (underlyingAsset, expiration)
    /eapi/v1/ticker                  24h volume, all symbols
    api.binance.com/api/v3/klines    historical spot bars (backtest only)

Designed for a 24/7 internal loop: `run_loop(on_row)` is awaited from app.py's
lifespan; `fetch_row()` is a one-shot convenience for scripts/smoke tests.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from typing import Awaitable, Callable, Optional

import httpx

from core.features import Contract, build_feature_row

log = logging.getLogger(__name__)

# Base URLs are overridable so a geo-blocked host (e.g. a US-region HF Space, which
# Binance answers with HTTP 451) can route through a proxy in an allowed region.
# See scripts/binance_proxy.py. Empty/unset env falls back to Binance directly.
EAPI = os.getenv("BINANCE_EAPI") or "https://eapi.binance.com"
SPOT_API = os.getenv("BINANCE_SPOT") or "https://api.binance.com"
FAPI = os.getenv("BINANCE_FAPI") or "https://fapi.binance.com"
PROXY_SECRET = os.getenv("BINANCE_PROXY_SECRET", "")


class BinanceOptionsFeed:
    """Polls Binance Options market data and emits feature rows.

    Parameters
    ----------
    underlying        option underlying symbol, e.g. "BTCUSDT"
    asset             OI endpoint's underlyingAsset, e.g. "BTC"
    skip_same_day     if True, skip the same-day (0DTE) expiry and use the next one,
                      holding it until it settles (then it rolls automatically). "Same
                      day" = current UTC date, matching Binance's YYMMDD expiry labels.
    min_expiry_hours  used only when skip_same_day is False: skip expiries closer than
                      this; falls back to the furthest expiry if all are too close
    ei_ttl            seconds to cache exchangeInfo (strikes/expiries are ~static intraday)
    """

    def __init__(self, underlying: str = "BTCUSDT", asset: str = "BTC",
                 skip_same_day: bool = False,
                 min_expiry_hours: float = 2.0, ei_ttl: float = 3600.0,
                 timeout: float = 15.0):
        self.underlying = underlying
        self.asset = asset
        self.skip_same_day = skip_same_day
        self.min_expiry_ms = min_expiry_hours * 3600 * 1000
        self.ei_ttl = ei_ttl
        self.timeout = timeout
        self._ei_cache: Optional[dict[str, dict]] = None  # symbol → meta
        self._ei_at: float = 0.0

    # ------------------------------------------------------------------ HTTP
    async def _get(self, client: httpx.AsyncClient, base: str, path: str,
                   params: Optional[dict] = None):
        headers = {"x-proxy-secret": PROXY_SECRET} if PROXY_SECRET else None
        r = await client.get(base + path, params=params, headers=headers,
                             timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    async def _symbol_meta(self, client: httpx.AsyncClient) -> dict[str, dict]:
        """{symbol: {side, strike, expiry_ms}} for our underlying, cached by TTL."""
        if self._ei_cache is not None and (time.time() - self._ei_at) < self.ei_ttl:
            return self._ei_cache
        ei = await self._get(client, EAPI, "/eapi/v1/exchangeInfo")
        meta: dict[str, dict] = {}
        for s in ei.get("optionSymbols", []):
            if s.get("underlying") != self.underlying:
                continue
            meta[s["symbol"]] = {
                "side": s["side"],                      # CALL | PUT
                "strike": float(s["strikePrice"]),
                "expiry_ms": int(s["expiryDate"]),
            }
        self._ei_cache, self._ei_at = meta, time.time()
        log.info("exchangeInfo: %d %s option symbols", len(meta), self.underlying)
        return meta

    def _choose_expiry(self, meta: dict[str, dict], now_ms: int) -> int:
        """Pick the working expiry.

        skip_same_day: nearest expiry on a *later* UTC date (skips 0DTE), held until
        it settles and rolls off. Otherwise: nearest expiry ≥ min_expiry_hours out.
        Both fall back to the furthest listed expiry.
        """
        exps = sorted({m["expiry_ms"] for m in meta.values()})
        if not exps:
            return 0
        if self.skip_same_day:
            today = datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc).date()
            future = [e for e in exps
                      if datetime.fromtimestamp(e / 1000, tz=timezone.utc).date() > today]
        else:
            future = [e for e in exps if e - now_ms >= self.min_expiry_ms]
        return future[0] if future else exps[-1]

    # -------------------------------------------------------------- snapshot
    async def fetch_snapshot(self, client: httpx.AsyncClient) -> dict:
        """Fetch + merge one snapshot: option chain + next-expiry IV + perp signals."""
        meta = await self._symbol_meta(client)
        now_ms = int(time.time() * 1000)
        exps = sorted({m["expiry_ms"] for m in meta.values()})
        expiry_ms = self._choose_expiry(meta, now_ms)
        expiry_yymmdd = datetime.fromtimestamp(
            expiry_ms / 1000, tz=timezone.utc).strftime("%y%m%d")

        # symbols in the chosen expiry
        chain_syms = {s for s, m in meta.items() if m["expiry_ms"] == expiry_ms}
        # Reference expiry for the IV term-structure slope: ~7d past the front, so the
        # slope reflects front-vs-term vol (adjacent dailies are nearly flat).
        term_gap_ms = 7 * 86400 * 1000
        next_ms = next((e for e in exps if e >= expiry_ms + term_gap_ms), None)
        if next_ms is None and exps and exps[-1] > expiry_ms:
            next_ms = exps[-1]

        # options chain (essential) + perp signals (best-effort) in parallel
        (index, mark, ticker, oi), perp = await asyncio.gather(
            asyncio.gather(
                self._get(client, EAPI, "/eapi/v1/index",
                          {"underlying": self.underlying}),
                self._get(client, EAPI, "/eapi/v1/mark"),
                self._get(client, EAPI, "/eapi/v1/ticker"),
                self._get(client, EAPI, "/eapi/v1/openInterest",
                          {"underlyingAsset": self.asset, "expiration": expiry_yymmdd}),
            ),
            self._fetch_perp(client),
        )

        spot = float(index["indexPrice"])
        mark_all = {m["symbol"]: m for m in mark}
        vol_by = {t["symbol"]: t for t in ticker if t["symbol"] in chain_syms}
        oi_by = {o["symbol"]: o for o in oi if o["symbol"] in chain_syms}

        contracts: list[Contract] = []
        for sym in chain_syms:
            mk = mark_all.get(sym, {})
            contracts.append({
                "side": meta[sym]["side"],
                "strike": meta[sym]["strike"],
                "mark_iv": _f(mk.get("markIV"), -1.0),
                "bid_iv": _f(mk.get("bidIV"), -1.0),
                "ask_iv": _f(mk.get("askIV"), -1.0),
                "delta": _f(mk.get("delta")),
                "theta": _f(mk.get("theta")),
                "vega": _f(mk.get("vega")),
                "gamma": _f(mk.get("gamma")),
                "mark_price": _f(mk.get("markPrice")),
                "oi": _f(oi_by.get(sym, {}).get("sumOpenInterest")),
                "oi_usd": _f(oi_by.get(sym, {}).get("sumOpenInterestUsd")),
                "volume": _f(vol_by.get(sym, {}).get("volume")),
            })

        next_atm_iv = (self._next_atm_iv(meta, mark_all, next_ms, spot)
                       if next_ms else None)

        return {"spot": spot, "ts_ms": now_ms, "contracts": contracts,
                "expiry": expiry_yymmdd, "next_atm_iv": next_atm_iv, "perp": perp}

    async def _fetch_perp(self, client: httpx.AsyncClient) -> dict:
        """Perpetual-futures sentiment/positioning — best-effort (never fatal)."""
        sym = self.underlying
        try:
            prem, oi, top, glob, taker = await asyncio.gather(
                self._get(client, FAPI, "/fapi/v1/premiumIndex", {"symbol": sym}),
                self._get(client, FAPI, "/fapi/v1/openInterest", {"symbol": sym}),
                self._get(client, FAPI, "/futures/data/topLongShortAccountRatio",
                          {"symbol": sym, "period": "5m", "limit": 1}),
                self._get(client, FAPI, "/futures/data/globalLongShortAccountRatio",
                          {"symbol": sym, "period": "5m", "limit": 1}),
                self._get(client, FAPI, "/futures/data/takerlongshortRatio",
                          {"symbol": sym, "period": "5m", "limit": 1}),
                return_exceptions=True,
            )
        except Exception:
            log.exception("perp fetch failed")
            return {}
        return self._parse_perp(prem, oi, top, glob, taker)

    @staticmethod
    def _parse_perp(prem, oi, top, glob, taker) -> dict:
        d: dict = {}
        if isinstance(prem, dict):
            try:
                mk, ix = float(prem["markPrice"]), float(prem["indexPrice"])
                d["funding_rate"] = float(prem.get("lastFundingRate", 0) or 0)
                d["perp_basis"] = (mk - ix) / ix if ix else 0.0
            except (KeyError, ValueError, TypeError):
                pass
        if isinstance(oi, dict):
            try:
                d["perp_oi"] = float(oi["openInterest"])
            except (KeyError, ValueError, TypeError):
                pass

        def _lsr(x):
            try:
                return float(x[0]["longShortRatio"])
            except (KeyError, ValueError, TypeError, IndexError):
                return None
        if isinstance(top, list) and (v := _lsr(top)) is not None:
            d["ls_ratio_top"] = v
        if isinstance(glob, list) and (v := _lsr(glob)) is not None:
            d["ls_ratio_global"] = v
        if isinstance(taker, list):
            try:
                d["taker_ratio"] = float(taker[0]["buySellRatio"])
            except (KeyError, ValueError, TypeError, IndexError):
                pass
        return d

    @staticmethod
    def _next_atm_iv(meta: dict, mark_all: dict, next_ms: int,
                     spot: float) -> Optional[float]:
        """Mean markIV of the call & put nearest spot in the next expiry."""
        syms = [s for s, m in meta.items() if m["expiry_ms"] == next_ms]

        def nearest(side: str) -> Optional[float]:
            cands = []
            for s in syms:
                if meta[s]["side"] != side:
                    continue
                iv = _f(mark_all.get(s, {}).get("markIV"), -1.0)
                if iv > 0:
                    cands.append((abs(meta[s]["strike"] - spot), iv))
            return min(cands)[1] if cands else None

        ivs = [v for v in (nearest("CALL"), nearest("PUT")) if v is not None]
        return round(sum(ivs) / len(ivs), 6) if ivs else None

    async def fetch_row(self) -> dict:
        """One-shot: open a client, fetch a snapshot, return the feature row."""
        async with httpx.AsyncClient() as client:
            snap = await self.fetch_snapshot(client)
        return build_feature_row(snap["spot"], snap["ts_ms"], snap["contracts"],
                                 expiry=snap["expiry"], next_atm_iv=snap["next_atm_iv"],
                                 perp=snap["perp"])

    # ------------------------------------------------------------------ loop
    async def run_loop(self, on_row: Callable[[dict, list], Awaitable | None],
                       interval: float = 60.0) -> None:
        """Poll every `interval`s forever, calling on_row(feature_row, chain) each cycle.

        The chain (list of merged contracts) is passed alongside the feature row so
        the paper trader can look up real option premiums. Network/parse errors are
        logged and skipped — the loop never dies on a single bad cycle. `on_row`
        may be sync or async.
        """
        async with httpx.AsyncClient() as client:
            while True:
                t0 = time.monotonic()
                try:
                    snap = await self.fetch_snapshot(client)
                    row = build_feature_row(snap["spot"], snap["ts_ms"],
                                            snap["contracts"], expiry=snap["expiry"],
                                            next_atm_iv=snap["next_atm_iv"],
                                            perp=snap["perp"])
                    res = on_row(row, snap["contracts"])
                    if asyncio.iscoroutine(res):
                        await res
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 451:
                        log.warning("Binance geo-blocked (451) from this host — point "
                                    "BINANCE_EAPI/BINANCE_FAPI/BINANCE_SPOT at a proxy "
                                    "in an allowed region (see scripts/binance_proxy.py)")
                    else:
                        log.warning("feed cycle HTTP %s for %s",
                                    e.response.status_code, e.request.url)
                except Exception:
                    log.exception("feed cycle failed")
                # keep cadence regardless of how long the cycle took
                await asyncio.sleep(max(1.0, interval - (time.monotonic() - t0)))


def _f(v, default: float = 0.0) -> float:
    try:
        x = float(v)
        return x if x == x else default  # NaN guard
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Historical spot bars — for the threshold backtest (no option history exists)
# ---------------------------------------------------------------------------
def fetch_klines(symbol: str = "BTCUSDT", interval: str = "1m",
                 limit: int = 1000, end_ms: Optional[int] = None) -> list[dict]:
    """Fetch up to `limit` spot klines. Sync (used by scripts, not the live loop)."""
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if end_ms is not None:
        params["endTime"] = end_ms
    r = httpx.get(SPOT_API + "/api/v3/klines", params=params, timeout=20.0)
    r.raise_for_status()
    return [{"open_time": k[0], "open": float(k[1]), "high": float(k[2]),
             "low": float(k[3]), "close": float(k[4]), "volume": float(k[5])}
            for k in r.json()]
