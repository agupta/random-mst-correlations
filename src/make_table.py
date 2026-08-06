"""Regenerate `data/kn-exact-table.csv`, the exact census of MST pair
correlations on K_n for 3 <= n <= 30.

Every column is an exact rational produced by `mst_exact.coalescent_moments`;
the two float columns are printed for reading only and are not used anywhere
in a decision.

    python3 src/make_table.py [nmax]
"""

from __future__ import annotations

import csv
import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mst_exact import coalescent_moments  # noqa: E402

HEADER = [
    "n", "p0", "p1", "p2", "p0_squared",
    "E_deg_squared", "E_L_n",
    "E_L_n_upper_bound", "E_L_n_lower_bound",
    "E_L_n_harmonic_lower_bound",
    "p1_over_p0sq_float", "p2_over_p0sq_float",
]


def harmonic_number(n: int) -> Fraction:
    return sum((Fraction(1, k) for k in range(1, n + 1)), Fraction(0))


def rows(nmax: int):
    for n in range(3, nmax + 1):
        m = coalescent_moments(n)
        p0 = Fraction(2, n)
        lo = Fraction((n - 1) * (n + 2), n * n)
        harmonic_lo = 1 + (harmonic_number(n) - 1) / n - Fraction(1, n * n)
        hi = Fraction((n - 1) * (5 * n + 6), 4 * n * n)
        yield [
            n, p0, m["p1"], m["p2"] if m["p2"] is not None else "",
            p0 * p0, m["ED2"], m["EL"], hi, lo, harmonic_lo,
            f"{float(m['p1'] / (p0 * p0)):.10f}",
            "" if m["p2"] is None else f"{float(m['p2'] / (p0 * p0)):.10f}",
        ]


def main() -> int:
    nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(here, "data", "kn-exact-table.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(HEADER)
        for r in rows(nmax):
            w.writerow([str(x) for x in r])
    print(f"wrote {out} for 3 <= n <= {nmax}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
