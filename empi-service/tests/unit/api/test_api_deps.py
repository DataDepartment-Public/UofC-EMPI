"""Unit tests for `src/api/deps.py` — mainly `get_backend`'s process-local
lock for the Parquet backend (see that module's docstring for why a real
FastAPI app needs one that a one-shot CLI never did).
"""

import pytest

from src.api import deps
from src.config import Settings


@pytest.fixture
def parquet_settings(tmp_path):
    s = Settings(project_root=tmp_path, index_backend="parquet")
    s.local_index_dir = tmp_path / "local_index"
    s.ensure_dirs()
    return s


@pytest.fixture
def sqlite_settings(tmp_path):
    s = Settings(project_root=tmp_path, index_backend="sqlite")
    s.db_path = tmp_path / "empi.db"
    s.ensure_dirs()
    return s


def test_get_backend_holds_lock_for_parquet_backend(parquet_settings):
    gen = deps.get_backend(parquet_settings)
    backend = next(gen)
    try:
        assert deps._PARQUET_BACKEND_LOCK.locked()
    finally:
        with pytest.raises(StopIteration):
            next(gen)
    assert not deps._PARQUET_BACKEND_LOCK.locked()
    backend.close()  # already closed by the generator's finally; idempotent


def test_get_backend_never_locks_for_sqlite_backend(sqlite_settings):
    gen = deps.get_backend(sqlite_settings)
    next(gen)
    assert not deps._PARQUET_BACKEND_LOCK.locked()
    with pytest.raises(StopIteration):
        next(gen)
    assert not deps._PARQUET_BACKEND_LOCK.locked()


def test_get_backend_releases_lock_even_on_exception(parquet_settings):
    gen = deps.get_backend(parquet_settings)
    next(gen)
    assert deps._PARQUET_BACKEND_LOCK.locked()
    with pytest.raises(ValueError):
        gen.throw(ValueError("simulated route handler failure"))
    assert not deps._PARQUET_BACKEND_LOCK.locked()
