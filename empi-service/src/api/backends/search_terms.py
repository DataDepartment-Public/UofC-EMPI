"""Shared free-text search tokenizing for the index backends.

The dashboard's one search box is matched against several *separate*
columns (`first_name`, `last_name`, `patid`, `mid`). Matching the raw query
string against each column independently means a full name never matches:
`first_name LIKE '%PATRICIA PATEL%'` is false, and so is the `last_name`
form, because no single column holds both words. A steward typing the name
in front of them got an empty result.

So the query is split on whitespace and **every token must match, but each
may match a different column of the same record** — "PATRICIA PATEL" hits
the record whose `first_name` is PATRICIA and `last_name` is PATEL. Tokens
are ANDed (narrowing, as a user expects of added words) while the columns
within a token are ORed. Order doesn't matter, so "PATEL PATRICIA" and a
"LAST, FIRST" paste both work.

The all-tokens-on-one-record part matters: ANDing across an entity's whole
member list instead would let one member supply "PATRICIA" and a different
one supply "PATEL", returning a cluster where nobody is named Patricia
Patel. Each backend therefore applies `search_tokens` inside a single
record's scope — one `entity_member` row for the registry, one side of the
pair for the review queue.

Kept here rather than in any one backend because all three
(`sql_backend`, `postgres_backend`, `parquet_backend`) must agree on what a
query means; a search that behaves differently on SQLite and Postgres is
the kind of drift `IndexBackend` exists to prevent.
"""

from __future__ import annotations


def search_tokens(search: str | None) -> list[str]:
    """Split a free-text search box query into match tokens.

    Returns `[]` for None/blank/whitespace-only input, which callers treat
    as "no search filter" — a box holding only spaces should not filter
    every row out.
    """
    if not search:
        return []
    return search.split()
