from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import requests

from src.utils.io_utils import ensure_dir, stable_cache_key

LOGGER = logging.getLogger(__name__)


class CachedHttpClient:
    def __init__(self, cache_dir: Path, max_retries: int = 5, backoff_sec: float = 1.5) -> None:
        self.cache_dir = cache_dir
        self.max_retries = max_retries
        self.backoff_sec = backoff_sec
        ensure_dir(cache_dir)

    def get_json(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        key = stable_cache_key({"url": url, "params": params})
        cache_path = self.cache_dir / f"{key}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))

        for i in range(self.max_retries):
            try:
                resp = requests.get(url, params=params, timeout=30)
                if resp.status_code >= 500:
                    raise requests.HTTPError(f"server error {resp.status_code}")
                resp.raise_for_status()
                payload = resp.json()
                cache_path.write_text(json.dumps(payload), encoding="utf-8")
                return payload
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
                wait = self.backoff_sec * (2**i)
                LOGGER.warning("HTTP retry %s/%s for %s params=%s err=%s", i + 1, self.max_retries, url, params, exc)
                time.sleep(wait)
        raise RuntimeError(f"failed request url={url} params={params}")
