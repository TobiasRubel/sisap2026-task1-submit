#!/usr/bin/env python3
"""Compare two directories of SISAP Task 1 result files.

    ./verify/compare_results.py <dir_a> <dir_b> [--gt dataset.h5] [--csv out.csv]

Files are matched by name; each holds an n x (k+1) int32 `knns` and an n x (k+1)
float32 `dists`, plus attributes. Reports, per config:

  knns_exact      knns arrays element-wise identical
  dists_bitwise   dists identical compared as raw int32 (so -0.0 and NaN cannot
                  masquerade as equal, which `==` would allow)
  mean_overlap    mean per-row |set(A) & set(B)| / k, self column excluded
  rows_diff       rows whose neighbour SET differs
  order_only      rows where the set matches but the order differs (a benign
                  class: it means only tie-breaking moved)
  max_abs_ddist   largest |a - b| over positions where the ids agree
  recall          recall@k against ground truth, by eval.py's formula, if --gt

buildtime/querytime are wall-clock and are always excluded. The four string
attributes are required to match exactly.
"""
import argparse
import csv
import os
import sys

import h5py
import numpy as np

STR_ATTRS = ("algo", "dataset", "task", "params")


def _attr(f, name):
    v = f.attrs.get(name)
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.ndarray):
        v = v[0]
        return v.decode() if isinstance(v, bytes) else str(v)
    return None if v is None else str(v)


def compare_pair(pa, pb, gt=None):
    with h5py.File(pa, "r") as fa, h5py.File(pb, "r") as fb:
        ka, kb = fa["knns"][:], fb["knns"][:]
        da, db = fa["dists"][:], fb["dists"][:]
        attrs_match = all(_attr(fa, a) == _attr(fb, a) for a in STR_ATTRS)
        params = _attr(fa, "params") or ""

    row = {
        "file": os.path.basename(pa),
        "params": params,
        "shape_match": ka.shape == kb.shape,
        "attrs_match": attrs_match,
    }
    if ka.shape != kb.shape:
        return row

    row["knns_exact"] = bool(np.array_equal(ka, kb))
    row["dists_bitwise"] = bool(
        np.array_equal(da.view(np.int32), db.view(np.int32))
    )

    # Column 0 is the point's own id (the self-match the grader expects); the
    # neighbours proper are columns 1..k.
    na, nb = ka[:, 1:], kb[:, 1:]
    k = na.shape[1]
    sa = np.sort(na, axis=1)
    sb = np.sort(nb, axis=1)
    set_equal = np.all(sa == sb, axis=1)
    row["rows_diff"] = int((~set_equal).sum())
    row["order_only"] = int((set_equal & np.any(na != nb, axis=1)).sum())

    if row["rows_diff"] == 0:
        row["mean_overlap"] = 1.0
        row["min_overlap"] = 1.0
    else:
        # Only the rows whose sets actually differ need the (slow) per-row
        # intersection; every other row is 1.0 by definition. At 6.35M rows with
        # a handful differing, looping over all of them would take hours.
        overlaps = np.ones(na.shape[0], dtype=np.float64)
        for i in np.flatnonzero(~set_equal):
            overlaps[i] = len(np.intersect1d(na[i], nb[i], assume_unique=False)) / k
        row["mean_overlap"] = float(overlaps.mean())
        row["min_overlap"] = float(overlaps.min())

    agree = ka == kb
    row["max_abs_ddist"] = float(np.abs(da[agree] - db[agree]).max()) if agree.any() else 0.0

    if gt is not None:
        row["recall_a"] = recall_at_k(ka, gt)
        row["recall_b"] = recall_at_k(kb, gt)
    return row


def recall_at_k(knns, gt):
    """eval.py's formula: intersect the first k columns, self included."""
    k = gt.shape[1]
    n = min(knns.shape[0], gt.shape[0])
    hits = 0
    for i in range(n):
        hits += len(np.intersect1d(knns[i, :k], gt[i, :k]))
    return hits / (n * k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir_a")
    ap.add_argument("dir_b")
    ap.add_argument("--gt", help="dataset .h5 with allknn/knns ground truth")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--csv")
    ap.add_argument("--max-ddist", type=float, default=1e-6,
                    help="fail if any |delta dist| exceeds this. The run-to-run "
                         "baseline is ~1.2e-07 (one ulp); anything larger means a "
                         "computed value changed, not just a store order.")
    args = ap.parse_args()

    gt = None
    if args.gt:
        with h5py.File(args.gt, "r") as f:
            gt = f["allknn"]["knns"][:, : args.k]

    names = sorted(
        set(os.listdir(args.dir_a)) & set(os.listdir(args.dir_b))
    )
    names = [n for n in names if n.endswith(".h5")]
    if not names:
        sys.exit("no result files in common between the two directories")

    rows = [
        compare_pair(os.path.join(args.dir_a, n), os.path.join(args.dir_b, n), gt)
        for n in names
    ]

    knns_ok = all(r.get("knns_exact") for r in rows)
    dists_ok = all(r.get("dists_bitwise") for r in rows)
    for r in rows:
        flag = "OK  " if r.get("knns_exact") else "DIFF"
        line = (
            f"{flag} {r['file']:<52} knns={r.get('knns_exact')!s:<5} "
            f"dists={r.get('dists_bitwise')!s:<5} rows_diff={r.get('rows_diff')} "
            f"order_only={r.get('order_only')} overlap={r.get('mean_overlap'):.6f}"
        )
        if "recall_a" in r:
            line += f" recall={r['recall_a']:.4f}/{r['recall_b']:.4f}"
        print(line)

    if args.csv:
        keys = sorted({k for r in rows for k in r})
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}")

    # knns is the graded output and must match exactly. dists is expected to
    # vary by ~1 ulp between any two runs, including two runs of the SAME
    # binary; see README. So a dists difference alone is not a failure — but a
    # LARGE one is, because it means a value changed rather than a store order.
    attrs_ok = all(r.get("attrs_match") for r in rows)
    worst = max((r.get("max_abs_ddist", 0.0) for r in rows), default=0.0)
    ddist_ok = worst <= args.max_ddist

    print(f"\n{len(rows)} configs compared")
    print(f"  knns  (graded output): {'IDENTICAL' if knns_ok else 'DIFFERENCES — INVESTIGATE'}")
    print(f"  attrs (algo/dataset/task/params): {'match' if attrs_ok else 'DIFFER — INVESTIGATE'}")
    print(
        f"  dists (advisory)     : {'identical' if dists_ok else 'differ (expected: ~1 ulp, run-to-run)'}"
    )
    print(
        f"  max |delta dist|     : {worst:.3e}"
        + ("" if ddist_ok else f"  EXCEEDS {args.max_ddist:.1e} — a computed value changed")
    )
    return 0 if (knns_ok and attrs_ok and ddist_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
