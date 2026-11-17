"""Command-line interface: ``errordep panel.csv --gold gold``."""
from __future__ import annotations

import argparse
import csv
import json
import sys

from .core import Panel

_TRUE = {"1", "true", "t", "yes", "y"}
_FALSE = {"0", "false", "f", "no", "n"}


def _flag(value: str, column: str, row: int) -> bool:
    v = value.strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    raise SystemExit(f"row {row}, column {column!r}: expected 0/1 or true/false, got {value!r}")


def load_panel(path, gold=None, as_correct=False, as_wrong=False, checkers=None,
               cluster_col=None, ignore=()) -> Panel:
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        rows = list(reader)
    if not rows:
        raise SystemExit("input has no data rows")
    special = {c for c in (gold, cluster_col) if c} | set(ignore)
    cols = list(checkers) if checkers else [c for c in header if c not in special]
    missing = [c for c in cols + [c for c in special if c] if c not in header]
    if missing:
        raise SystemExit(f"column(s) not found in {path}: {', '.join(missing)}")
    if gold is None and not (as_correct or as_wrong):
        raise SystemExit("give --gold COLUMN, or say the columns are --as-correct or --as-wrong flags")
    wrong = []
    for n, r in enumerate(rows, start=2):
        if gold is not None:
            wrong.append([r[c].strip() != r[gold].strip() for c in cols])
        elif as_correct:
            wrong.append([not _flag(r[c], c, n) for c in cols])
        else:
            wrong.append([_flag(r[c], c, n) for c in cols])
    clusters = [r[cluster_col] for r in rows] if cluster_col else None
    return Panel(wrong, cols, clusters)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="errordep",
        description="Test whether a panel of LLM judges/verifiers makes independent errors.",
    )
    ap.add_argument("csv", help="CSV with one row per item and one column per checker")
    ap.add_argument("--gold", help="column holding the ground truth; checker columns are their answers")
    ap.add_argument("--as-correct", action="store_true", help="checker columns are 1/0 'was correct' flags")
    ap.add_argument("--as-wrong", action="store_true", help="checker columns are 1/0 'was wrong' flags")
    ap.add_argument("--checkers", help="comma-separated checker columns (default: all other columns)")
    ap.add_argument("--cluster-col", help="column with a group id per item; bootstrap resamples groups")
    ap.add_argument("--ignore", default="", help="comma-separated columns to skip (for example an id column)")
    ap.add_argument("--permutations", type=int, default=2000)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true", help="print JSON instead of text")
    a = ap.parse_args(argv)

    panel = load_panel(
        a.csv, gold=a.gold, as_correct=a.as_correct, as_wrong=a.as_wrong,
        checkers=a.checkers.split(",") if a.checkers else None,
        cluster_col=a.cluster_col, ignore=[c for c in a.ignore.split(",") if c],
    )
    rep = panel.report(a.permutations, a.bootstrap, a.seed)
    print(json.dumps(rep.to_dict(), indent=2) if a.json else rep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
