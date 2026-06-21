"""Unit tests for ComparisonRegistry — ordering invariants + mutation semantics."""

from __future__ import annotations

import pytest

from models.common.fs_base import ComparisonRegistry, ComparisonSpec


def _spec(name: str) -> ComparisonSpec:
    return ComparisonSpec(name=name, builder=lambda n=name: {"output_column_name": n})


def test_registry_preserves_declaration_order():
    reg = ComparisonRegistry([_spec("a"), _spec("b"), _spec("c")])
    assert reg.names() == ["a", "b", "c"]


def test_registry_rejects_duplicate_names():
    with pytest.raises(ValueError, match="duplicate"):
        ComparisonRegistry([_spec("a"), _spec("b"), _spec("a")])


def test_with_added_appends_by_default_and_returns_new_registry():
    original = ComparisonRegistry([_spec("a"), _spec("b")])
    extended = original.with_added(_spec("c"))
    assert original.names() == ["a", "b"]  # unchanged
    assert extended.names() == ["a", "b", "c"]


def test_with_added_at_position_inserts():
    reg = ComparisonRegistry([_spec("a"), _spec("c")])
    extended = reg.with_added(_spec("b"), position=1)
    assert extended.names() == ["a", "b", "c"]


def test_with_removed_returns_new_registry_without_name():
    reg = ComparisonRegistry([_spec("a"), _spec("b"), _spec("c")])
    reduced = reg.with_removed("b")
    assert reg.names() == ["a", "b", "c"]  # unchanged
    assert reduced.names() == ["a", "c"]


def test_with_removed_unknown_name_raises():
    reg = ComparisonRegistry([_spec("a")])
    with pytest.raises(KeyError, match="b"):
        reg.with_removed("b")


def test_with_replaced_preserves_position():
    reg = ComparisonRegistry([_spec("a"), _spec("b"), _spec("c")])
    replaced = reg.with_replaced("b", ComparisonSpec("b", lambda: {"v": 42}))
    assert replaced.names() == ["a", "b", "c"]
    assert replaced.get("b").builder() == {"v": 42}


def test_with_replaced_unknown_name_raises():
    reg = ComparisonRegistry([_spec("a")])
    with pytest.raises(KeyError):
        reg.with_replaced("missing", _spec("x"))


def test_build_all_calls_each_builder_lazily():
    calls: list[str] = []

    def builder(name: str) -> dict:
        calls.append(name)
        return {"output_column_name": name}

    reg = ComparisonRegistry([
        ComparisonSpec("a", lambda: builder("a")),
        ComparisonSpec("b", lambda: builder("b")),
    ])
    assert calls == []  # not invoked at registry construction
    dicts = reg.build_all()
    assert calls == ["a", "b"]
    assert dicts == [{"output_column_name": "a"}, {"output_column_name": "b"}]


def test_contains_and_get():
    reg = ComparisonRegistry([_spec("a"), _spec("b")])
    assert "a" in reg
    assert "z" not in reg
    assert reg.get("a").name == "a"
    with pytest.raises(KeyError):
        reg.get("z")
