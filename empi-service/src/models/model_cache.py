"""In-memory cache for the FS and ML matcher artifacts.

Today neither matcher is cached: `FSMatcher.load_settings` (a `json.load`)
and `registry.load_model_artifact` (a `joblib.load`) both run fresh on every
pipeline run and every incremental `/records/score` call — see
`src/pipeline.py` and `src/api/ingest/incremental.py`. That means a promoted
model already takes effect on the very next call with zero code changes.
This module exists to avoid re-reading/re-deserializing the same artifact on
every single call once a deployment sees real traffic, without reintroducing
the staleness that caching usually costs you:

- Cache key is `(path, mtime)`, not just `path` — the moment a promoted
  model's file changes underneath a cache entry, the next lookup detects the
  mtime mismatch and reloads. Self-invalidating; no explicit action required
  for correctness.
- `invalidate()` exists anyway so `POST /admin/models/reload` (see
  `src/api/routers/admin.py`) can force an eager reload right after a
  promotion — an explicit, synchronous, auditable "it's live now" moment
  instead of waiting for whichever request happens to notice first.

Thread-safe: FastAPI's sync `def` routes (every route in this service) run
on Starlette's threadpool, so concurrent requests can race on the same
cache entry without a lock.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable

_lock = threading.Lock()
_cache: dict[str, tuple[Path, float, Any]] = {}


def get_or_load(key: str, path: Path, loader: Callable[[Path], Any]) -> Any:
    """Return the cached object for `key` if `path` hasn't changed on disk
    since it was cached (same path, same mtime); otherwise load it via
    `loader(path)`, cache it under `key`, and return it."""
    path = Path(path)
    mtime = path.stat().st_mtime
    with _lock:
        cached = _cache.get(key)
        if cached is not None and cached[0] == path and cached[1] == mtime:
            return cached[2]
    obj = loader(path)
    with _lock:
        _cache[key] = (path, mtime, obj)
    return obj


def invalidate(key: str | None = None) -> list[str]:
    """Drop cached entries so the next `get_or_load` call reloads from disk.
    With no key, drops everything. Returns the keys that were removed."""
    with _lock:
        if key is None:
            removed = list(_cache.keys())
            _cache.clear()
            return removed
        if key in _cache:
            del _cache[key]
            return [key]
        return []


def status() -> dict[str, dict[str, Any]]:
    """What's currently cached — path + mtime per key, for `/admin/models/status`."""
    with _lock:
        return {key: {"path": str(p), "mtime": m} for key, (p, m, _obj) in _cache.items()}


__all__ = ["get_or_load", "invalidate", "status"]
