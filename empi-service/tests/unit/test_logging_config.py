"""Unit tests for src/logging_config.configure_logging.

Covers level selection, handler idempotency, and optional file output — the
behaviors driven by the EMPI_LOG_* settings.
"""

import logging

from src.config import Settings, configure_logging


def test_level_from_settings():
    settings = Settings(log_level="WARNING")
    root = configure_logging(settings, force=True)
    assert root.level == logging.WARNING


def test_explicit_level_overrides_settings():
    settings = Settings(log_level="ERROR")
    root = configure_logging(settings, level="DEBUG", force=True)
    assert root.level == logging.DEBUG


def test_invalid_level_falls_back_to_info():
    settings = Settings(log_level="NOPE")
    root = configure_logging(settings, force=True)
    assert root.level == logging.INFO


def test_force_does_not_stack_handlers():
    settings = Settings()
    configure_logging(settings, force=True)
    first = len(logging.getLogger().handlers)
    configure_logging(settings, force=True)
    assert len(logging.getLogger().handlers) == first


def test_file_handler_writes_log(tmp_path):
    settings = Settings(log_to_file=True, log_dir=tmp_path, log_file="t.log")
    configure_logging(settings, force=True)
    logging.getLogger("eMPI.test").warning("hello")
    for handler in logging.getLogger().handlers:
        handler.flush()
    log_file = tmp_path / "t.log"
    assert log_file.exists()
    assert "hello" in log_file.read_text()
    # Reset so the open file handler does not leak into other tests.
    configure_logging(Settings(), force=True)
