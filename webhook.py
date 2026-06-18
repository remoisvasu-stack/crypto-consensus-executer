"""webhook.py — Fire-and-forget signed POST helper for the live signal path.

Never raises into the collector loop: a transient brain outage just drops that
tick (acceptable — the model tolerates collection gaps, and the row is still
persisted to the HF dataset independently).
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

log = logging.getLogger("collector.webhook")


def post_json(url: str, payload: dict[str, Any], *, token: Optional[str] = None,
              header: str = "X-Webhook-Token", timeout: float = 10.0) -> bool:
    if not url:
        return False
    headers = {header: token} if token else None
    try:
        r = httpx.post(url, json=payload, headers=headers, timeout=timeout)
        if r.status_code >= 300:
            log.warning("POST %s -> %s %s", url, r.status_code, r.text[:200])
            return False
        return True
    except Exception as e:
        log.warning("POST %s failed: %s", url, e)
        return False
