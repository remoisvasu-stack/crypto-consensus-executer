"""core/features.py — Build a per-minute BTC option-chain feature row.

Pure, I/O-free transforms: given already-fetched Binance Options payloads (merged
into a per-contract list) plus the spot index price, produce one feature dict that
mirrors the role the NIFTY option-chain analytics played in the original system.

The collector (`core/binance_feed.py`) does the HTTP and merges raw endpoints into
the `contracts` list this module consumes, so everything here is unit-testable with
canned JSON.

Feature philosophy (NIFTY → BTC mapping):
    Vix (B7)          → atm_iv          (annualized ATM implied vol; the vol gate)
    OI PCR + bands    → oi_pcr[_Npct]   (put/call open-interest ratios by strike band)
    Volume Ratio      → vol_pcr
    IV PCR            → iv_pcr, iv_skew
    Delta/Theta/Vega  → *_pcr           (OI-weighted Greek put/call ratios)
    SR 3pt/4pt        → sr_support/resistance_dist (max-OI strikes vs spot)
    Net Dir Score     → net_dir_score   (composite call-vs-put tilt, ~[-1, 1])

Levels that are non-stationary (spot, total_oi, total_vol) are kept in the row for
analytics/dashboard but excluded from FEATURE_COLUMNS so the model trains only on
stationary ratios + IV + the engineered returns added later in consensus.py.
"""
from __future__ import annotations

import math
from typing import Optional, TypedDict

# Strike bands (fraction of spot) for windowed PCR / IV aggregates.
PCR_BANDS = (0.02, 0.04, 0.10)
SKEW_BAND = 0.05   # OTM put vs OTM call offset for iv_skew / iv_wing
_EPS = 1e-9
_RATIO_CLAMP = 100.0  # cap ratios so a near-empty denominator can't produce inf

# Stationary option-derived features safe to feed the model. Engineered price
# features (ret_*, vol_*, tod_*, iv_chg_5m) are appended later over the buffer.
FEATURE_COLUMNS = [
    "atm_iv",
    "oi_pcr", "oi_pcr_2pct", "oi_pcr_4pct", "oi_pcr_10pct", "oi_pcr_usd",
    "vol_pcr",
    "iv_skew", "iv_wing",
    "delta_pcr", "theta_pcr", "vega_pcr",
    "sr_support_dist", "sr_resistance_dist",
    "net_dir_score",
    # Group B — options-native extras
    "gamma_tilt", "iv_term_slope", "max_pain_dist", "iv_bidask",
    # Group A — perp-futures sentiment / positioning
    "funding_rate", "perp_basis", "ls_ratio_top", "ls_ratio_global",
    "ls_divergence", "taker_ratio",
]


class Contract(TypedDict, total=False):
    """One option contract merged from mark + openInterest + ticker + meta."""
    side: str        # "CALL" | "PUT"
    strike: float
    mark_iv: float   # annualized implied vol; <=0 means invalid (Binance sentinel)
    bid_iv: float    # bid implied vol; <=0 means invalid
    ask_iv: float    # ask implied vol; <=0 means invalid
    delta: float
    theta: float
    vega: float
    gamma: float
    oi: float        # sumOpenInterest, in contracts
    oi_usd: float    # sumOpenInterestUsd
    volume: float    # 24h contracts traded
    mark_price: float


def _f(v, default: float = 0.0) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def _ratio(put_val: float, call_val: float) -> float:
    """put/call ratio, clamped so a tiny denominator can't blow up to inf."""
    if call_val <= _EPS:
        return _RATIO_CLAMP if put_val > _EPS else 1.0
    return min(put_val / call_val, _RATIO_CLAMP)


def _valid_iv(c: Contract) -> Optional[float]:
    iv = _f(c.get("mark_iv"), -1.0)
    return iv if iv > 0 else None


def build_feature_row(
    spot: float,
    ts_ms: int,
    contracts: list[Contract],
    *,
    expiry: Optional[str] = None,
    next_atm_iv: Optional[float] = None,
    perp: Optional[dict] = None,
) -> dict:
    """Collapse one option-chain snapshot into a single feature row.

    `contracts` should already be filtered to a single underlying + chosen expiry.
    `next_atm_iv` is the ATM IV of the *next* expiry (for the term-structure slope).
    `perp` is the perpetual-futures signal dict from BinanceOptionsFeed._fetch_perp.
    Returns a dict with identity fields, raw levels (for analytics), and every
    name in FEATURE_COLUMNS. Missing/degenerate inputs degrade to neutral values
    (ratios → 1.0, distances → 0.0) rather than raising.
    """
    calls = [c for c in contracts if c.get("side") == "CALL"]
    puts = [c for c in contracts if c.get("side") == "PUT"]

    row: dict = {
        "ts": int(ts_ms),
        "expiry": expiry,
        "spot": _f(spot),
        "n_contracts": len(contracts),
    }

    # ---- ATM implied vol (VIX replacement) -------------------------------
    row["atm_iv"] = _atm_iv(spot, calls, puts)

    # ---- Open-interest PCR, overall + by strike band ---------------------
    put_oi_all = sum(_f(c.get("oi")) for c in puts)
    call_oi_all = sum(_f(c.get("oi")) for c in calls)
    put_oi_usd = sum(_f(c.get("oi_usd")) for c in puts)
    call_oi_usd = sum(_f(c.get("oi_usd")) for c in calls)
    row["oi_pcr"] = _ratio(put_oi_all, call_oi_all)
    row["oi_pcr_usd"] = _ratio(put_oi_usd, call_oi_usd)
    for band in PCR_BANDS:
        lo, hi = spot * (1 - band), spot * (1 + band)
        p = sum(_f(c.get("oi")) for c in puts if lo <= _f(c.get("strike")) <= hi)
        c_ = sum(_f(c.get("oi")) for c in calls if lo <= _f(c.get("strike")) <= hi)
        row[f"oi_pcr_{int(band * 100)}pct"] = _ratio(p, c_)

    # ---- Volume PCR ------------------------------------------------------
    put_vol = sum(_f(c.get("volume")) for c in puts)
    call_vol = sum(_f(c.get("volume")) for c in calls)
    row["vol_pcr"] = _ratio(put_vol, call_vol)

    # ---- IV smile shape: skew (asymmetry) + wing convexity ---------------
    # Binance publishes ONE markIV per strike (the smile), so a put/call IV
    # ratio is structurally ~1.0 and useless. The information lives in the
    # smile's shape: downside-vs-upside skew and wing-vs-ATM convexity.
    row["iv_skew"], row["iv_wing"] = _iv_shape(spot, calls, puts, row["atm_iv"])

    # ---- OI-weighted Greek PCRs ------------------------------------------
    row["delta_pcr"] = _ratio(
        abs(sum(_f(c.get("delta")) * _f(c.get("oi")) for c in puts)),
        abs(sum(_f(c.get("delta")) * _f(c.get("oi")) for c in calls)))
    row["theta_pcr"] = _ratio(
        abs(sum(_f(c.get("theta")) * _f(c.get("oi")) for c in puts)),
        abs(sum(_f(c.get("theta")) * _f(c.get("oi")) for c in calls)))
    row["vega_pcr"] = _ratio(
        abs(sum(_f(c.get("vega")) * _f(c.get("oi")) for c in puts)),
        abs(sum(_f(c.get("vega")) * _f(c.get("oi")) for c in calls)))

    # ---- Support / resistance from max-OI strikes ------------------------
    sup_strike, sup_oi = _max_oi_strike(puts, below=spot)
    res_strike, res_oi = _max_oi_strike(calls, above=spot)
    row["sr_support_dist"] = (sup_strike / spot - 1) if (sup_strike and spot) else 0.0
    row["sr_resistance_dist"] = (res_strike / spot - 1) if (res_strike and spot) else 0.0
    row["sr_support_oi"] = sup_oi
    row["sr_resistance_oi"] = res_oi

    # ---- Composite directional tilt, ~[-1, 1] (positive = bullish) -------
    oi_tilt = (call_oi_all - put_oi_all) / max(call_oi_all + put_oi_all, _EPS)
    vol_tilt = (call_vol - put_vol) / max(call_vol + put_vol, _EPS)
    row["net_dir_score"] = round((oi_tilt + vol_tilt) / 2, 6)

    # ---- Group B: options-native extras ----------------------------------
    row["gamma_tilt"] = _gamma_tilt(calls, puts)            # OI-weighted call vs put gamma
    row["max_pain_dist"] = _max_pain_dist(spot, calls, puts)  # max-pain strike vs spot
    row["iv_bidask"] = _iv_bidask(spot, calls, puts)        # ATM bid-ask IV spread
    row["iv_term_slope"] = (round(next_atm_iv - row["atm_iv"], 6)
                            if (next_atm_iv and row["atm_iv"] > 0) else 0.0)

    # ---- Group A: perp-futures sentiment / positioning -------------------
    p = perp or {}
    row["funding_rate"] = _f(p.get("funding_rate"), 0.0)
    row["perp_basis"] = _f(p.get("perp_basis"), 0.0)
    row["ls_ratio_top"] = _f(p.get("ls_ratio_top"), 1.0)
    row["ls_ratio_global"] = _f(p.get("ls_ratio_global"), 1.0)
    row["ls_divergence"] = round(row["ls_ratio_top"] - row["ls_ratio_global"], 6)
    row["taker_ratio"] = _f(p.get("taker_ratio"), 1.0)

    # ---- Non-stationary levels (analytics only) --------------------------
    row["total_oi"] = call_oi_all + put_oi_all
    row["total_vol"] = call_vol + put_vol
    row["perp_oi"] = _f(p.get("perp_oi"), 0.0)   # level → consensus adds perp_oi_chg5

    return row


def _atm_iv(spot: float, calls: list[Contract], puts: list[Contract]) -> float:
    """Mean markIV of the call & put nearest the spot strike. 0.0 if unavailable."""
    def nearest(side: list[Contract]) -> Optional[float]:
        cands = [(abs(_f(c.get("strike")) - spot), iv)
                 for c in side if (iv := _valid_iv(c))]
        return min(cands)[1] if cands else None

    ivs = [v for v in (nearest(calls), nearest(puts)) if v is not None]
    return round(sum(ivs) / len(ivs), 6) if ivs else 0.0


def _iv_at(side: list[Contract], target: float) -> Optional[float]:
    """markIV of the valid-IV contract whose strike is nearest `target`."""
    cands = [(abs(_f(c.get("strike")) - target), iv)
             for c in side if (iv := _valid_iv(c))]
    return min(cands)[1] if cands else None


def _iv_shape(spot: float, calls: list[Contract], puts: list[Contract],
              atm_iv: float) -> tuple[float, float]:
    """Return (skew, wing) of the vol smile.

    skew = OTM-put IV − OTM-call IV at ~±SKEW_BAND (positive = downside vol premium).
    wing = mean(OTM-put IV, OTM-call IV) − ATM IV (smile convexity / tail premium).
    """
    put_wing = _iv_at(puts, spot * (1 - SKEW_BAND))
    call_wing = _iv_at(calls, spot * (1 + SKEW_BAND))
    skew = (round(put_wing - call_wing, 6)
            if (put_wing is not None and call_wing is not None) else 0.0)
    wings = [v for v in (put_wing, call_wing) if v is not None]
    wing = (round(sum(wings) / len(wings) - atm_iv, 6)
            if (wings and atm_iv > 0) else 0.0)
    return skew, wing


def _gamma_tilt(calls: list[Contract], puts: list[Contract]) -> float:
    """OI-weighted call-vs-put gamma, in [-1, 1]. Positive = call gamma dominates."""
    cg = sum(_f(c.get("gamma")) * _f(c.get("oi")) for c in calls)
    pg = sum(_f(c.get("gamma")) * _f(c.get("oi")) for c in puts)
    tot = cg + pg
    return round((cg - pg) / tot, 6) if tot > _EPS else 0.0


def _max_pain_dist(spot: float, calls: list[Contract],
                   puts: list[Contract]) -> float:
    """Distance of the max-pain strike from spot (the strike that minimizes total
    option-holder intrinsic value at expiry). Returns max_pain/spot - 1."""
    strikes = sorted({_f(c.get("strike")) for c in (calls + puts)
                      if _f(c.get("strike")) > 0})
    if not strikes or spot <= 0:
        return 0.0
    best_k, best_pain = strikes[0], None
    for k in strikes:
        pain = (sum(_f(c.get("oi")) * max(k - _f(c.get("strike")), 0.0) for c in calls)
                + sum(_f(c.get("oi")) * max(_f(c.get("strike")) - k, 0.0) for c in puts))
        if best_pain is None or pain < best_pain:
            best_pain, best_k = pain, k
    return round(best_k / spot - 1, 6)


def _iv_bidask(spot: float, calls: list[Contract], puts: list[Contract]) -> float:
    """ATM bid-ask IV spread (ask_iv - bid_iv), averaged over the nearest call & put
    with valid quotes. A liquidity / uncertainty proxy."""
    def nearest_spread(side: list[Contract]) -> Optional[float]:
        cands = []
        for c in side:
            b, a = _f(c.get("bid_iv"), -1.0), _f(c.get("ask_iv"), -1.0)
            if b > 0 and a > 0:
                cands.append((abs(_f(c.get("strike")) - spot), a - b))
        return min(cands)[1] if cands else None

    sp = [s for s in (nearest_spread(calls), nearest_spread(puts)) if s is not None]
    return round(sum(sp) / len(sp), 6) if sp else 0.0


def _max_oi_strike(side: list[Contract], *, below: float = None,
                   above: float = None) -> tuple[Optional[float], float]:
    """Strike with the largest OI on one side, optionally constrained vs spot."""
    best_strike, best_oi = None, 0.0
    for c in side:
        k, oi = _f(c.get("strike")), _f(c.get("oi"))
        if below is not None and k >= below:
            continue
        if above is not None and k <= above:
            continue
        if oi > best_oi:
            best_strike, best_oi = k, oi
    return best_strike, best_oi
