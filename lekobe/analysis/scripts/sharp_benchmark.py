from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

PRIMARY_BOOKS = ("pinnacle", "fanduel", "betonline")
DK_BOOK = "draftkings"


def book_family(bookmaker: object) -> str:
    book = str(bookmaker).lower().strip()
    if book == "pinnacle":
        return "pinnacle"
    if book == "fanduel":
        return "fanduel"
    if "betonline" in book:
        return "betonline"
    if book == "draftkings":
        return "draftkings"
    return book


def consensus_from_rows(
    rows: pd.DataFrame,
    value_col: str,
    bookmaker_col: str = "bookmaker",
    sort_cols: Iterable[str] | None = None,
) -> dict[str, object]:
    """Average bookmaker-level no-vig prices after each book has been de-vigged."""
    if rows is None or rows.empty or value_col not in rows:
        return _missing()

    data = rows.copy()
    data["_book_family"] = data[bookmaker_col].map(book_family)
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    data = data[data[value_col].notna()].copy()
    if data.empty:
        return _missing()

    sort_by = [c for c in (sort_cols or []) if c in data.columns]
    if sort_by:
        data = data.sort_values(sort_by)
    dedup = data.drop_duplicates("_book_family", keep="first")

    primary_values: list[float] = []
    primary_books: list[str] = []
    for family in PRIMARY_BOOKS:
        hit = dedup[dedup["_book_family"].eq(family)]
        if not hit.empty:
            primary_values.append(float(hit.iloc[0][value_col]))
            primary_books.append(family)

    if primary_values:
        count = len(primary_values)
        if count == 3:
            source = "pin_fd_betonline_consensus_3"
            confidence = "primary_3_book"
        elif count == 2:
            source = "pin_fd_betonline_consensus_2"
            confidence = "primary_2_book"
        else:
            source = f"primary_single_{primary_books[0]}"
            confidence = "primary_1_book"
        return {
            "value": float(np.mean(primary_values)),
            "source": source,
            "books": ",".join(primary_books),
            "book_count": count,
            "confidence": confidence,
        }

    non_dk = dedup[~dedup["_book_family"].eq(DK_BOOK)]
    if not non_dk.empty:
        books = sorted(non_dk["_book_family"].astype(str).unique())
        return {
            "value": float(non_dk[value_col].mean()),
            "source": "broader_consensus_ex_dk",
            "books": ",".join(books),
            "book_count": int(len(books)),
            "confidence": "lower_broader_consensus",
        }

    dk = dedup[dedup["_book_family"].eq(DK_BOOK)]
    if not dk.empty:
        return {
            "value": float(dk.iloc[0][value_col]),
            "source": "draftkings_fallback",
            "books": DK_BOOK,
            "book_count": 1,
            "confidence": "lower_dk",
        }

    return _missing()


def _missing() -> dict[str, object]:
    return {
        "value": np.nan,
        "source": "missing",
        "books": "",
        "book_count": 0,
        "confidence": "missing",
    }
