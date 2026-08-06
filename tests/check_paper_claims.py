#!/usr/bin/env python3
"""Verification suite for the claims of `paper/main.tex`.

Everything here is exact: Python `int` and `fractions.Fraction` only, standard
library only, no floating point in any decision.

Each block states which claim of the paper it touches.  Two blocks deserve
special mention.

*   Block 2 is the bridge that the historical iteration-96 certificate did not
    contain (see `evidence/PROVENANCE.md`): it checks the accepted-merger
    coalescent of Lemma 5.1 against the MST measure computed from its own
    definition, by brute-force enumeration of edge orderings on K_4 and by
    exact order-pattern summation on K_5.  Without this block the coalescent is
    an assumption rather than a verified model.

*   Block 5 checks E[L_n] by a route (the exact G(n,p) connectivity polynomial
    plus Kruskal's area identity) that knows nothing about mergers, degrees or
    the coalescent, and then against Gamarnik's published table, which is the
    only numerical anchor outside this work.

Usage:
    python3 tests/check_paper_claims.py            # fast suite, `make check`
    python3 tests/check_paper_claims.py --full     # wider ranges, `make check-full`

Exit status is 0 if and only if every check passed.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from fractions import Fraction
from functools import lru_cache
from itertools import combinations, combinations_with_replacement, permutations

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from mst_exact import (  # noqa: E402
    bundle_family,
    coalescent_moments,
    complete_graph_edges,
    expected_mst_length_poly,
    hierarchy_HJ,
    hierarchy_shapes,
    hub_family_finite,
    maxlaw_family,
    maxlaw_family_by_integration,
    mst_marginals,
    mst_pair_probability,
    partition_lhs_rhs,
    partitions,
    perm_mst_probabilities,
)

FAILURES: list[str] = []
PASSES = 0
REPORTED_FAILURES = 0


def check(condition, label: str) -> None:
    global PASSES
    if condition:
        PASSES += 1
    else:
        FAILURES.append(label)


def report(line: str) -> None:
    global REPORTED_FAILURES
    if line.startswith("PASS block"):
        new_failures = len(FAILURES) - REPORTED_FAILURES
        if new_failures:
            line = "FAIL" + line[4:] + f" ({new_failures} new failure(s))"
        REPORTED_FAILURES = len(FAILURES)
    print(line, flush=True)


# The Lyons-Peres-Schramm multigraph: K_4 with the two disjoint edges 01 and 23
# each replaced by three parallel copies.  Index 0 is a marked copy of the first
# bundle, index 3 a marked copy of the second.
LPS_EDGES = [(0, 1)] * 3 + [(2, 3)] * 3 + [(0, 2), (0, 3), (1, 2), (1, 3)]

# K_5 minus edges 24 and 34.  Mark 04 and 23.
SIMPLE_COUNTEREXAMPLE_EDGES = [
    (0, 1), (0, 2), (0, 3), (0, 4),
    (1, 2), (1, 3), (1, 4), (2, 3),
]


def hub_witness_edges(s: int):
    """The simple hub graph G_s: marked edges 01 and 23, with every hub
    adjacent to each of the four wing vertices."""
    return (
        [(0, 1), (2, 3)]
        + [(wing, hub) for hub in range(4, 4 + s) for wing in range(4)]
    )


def harmonic_number(n: int) -> Fraction:
    return sum((Fraction(1, k) for k in range(1, n + 1)), Fraction(0))


# ---------------------------------------------------------------------------
# block 1: the MST measure from its own definition
# ---------------------------------------------------------------------------

def block1_measure_self_tests(full: bool) -> None:
    # K_4 by brute force over all 6! orderings, and by order-pattern summation.
    edges4 = complete_graph_edges(4)
    marg_perm, pair_perm = perm_mst_probabilities(4, edges4)
    marg_pat = mst_marginals(4, edges4)
    check(marg_perm == marg_pat, "K_4 marginals: ordering route != pattern route")
    check(all(p == Fraction(1, 2) for p in marg_perm), "K_4: p_0 != 2/n")
    check(sum(marg_perm) == 3, "K_4: sum_e p_e != n-1")
    for (i, j), value in pair_perm.items():
        check(value == mst_pair_probability(4, edges4, i, j),
              f"K_4 pair {(i, j)}: ordering route != pattern route")

    # K_5 by the pattern route.
    edges5 = complete_graph_edges(5)
    marg5 = mst_marginals(5, edges5)
    check(all(p == Fraction(2, 5) for p in marg5), "K_5: p_0 != 2/n")
    check(sum(marg5) == 4, "K_5: sum_e p_e != n-1")

    # The LPS multigraph: exact published values, and positive correlation.
    lps_marg = mst_marginals(4, LPS_EDGES)
    pe, pf = lps_marg[0], lps_marg[3]
    pef = mst_pair_probability(4, LPS_EDGES, 0, 3)
    check(pe == pf == Fraction(331, 1260), "LPS: p_e != 331/1260")
    check(pef == Fraction(109, 1575), "LPS: p_ef != 109/1575")
    check(pef / (pe * pf) == Fraction(109872, 109561), "LPS: ratio != 109872/109561")
    check(pef > pe * pf, "LPS control: unconditional p-NC was NOT refuted")
    check(pef <= 8 * pe * pf, "LPS: Theorem A violated")
    check(sum(lps_marg) == 3, "LPS: sum_e p_e != n-1")
    # Two parallel copies of one bundle are never jointly in the tree.
    check(mst_pair_probability(4, LPS_EDGES, 0, 1) == 0,
          "LPS: two parallel copies have nonzero joint probability")

    # Proposition 4.3: a simple-graph counterexample, by both direct routes.
    simple_marg = mst_marginals(5, SIMPLE_COUNTEREXAMPLE_EDGES)
    simple_pef = mst_pair_probability(5, SIMPLE_COUNTEREXAMPLE_EDGES, 3, 7)
    simple_perm_marg, simple_perm_pair = perm_mst_probabilities(
        5, SIMPLE_COUNTEREXAMPLE_EDGES)
    check(simple_marg == simple_perm_marg,
          "simple counterexample: marginal routes disagree")
    check(simple_pef == simple_perm_pair[(3, 7)],
          "simple counterexample: pair routes disagree")
    pe, pf = simple_marg[3], simple_marg[7]
    check((pe, pf, simple_pef) ==
          (Fraction(7, 12), Fraction(69, 140), Fraction(145, 504)),
          "simple counterexample: exact probabilities differ from paper")
    check(simple_pef / (pe * pf) == Fraction(1450, 1449),
          "simple counterexample: ratio is not 1450/1449")

    # The stronger simple hub witnesses.  The s=2 graph is small enough for
    # full 10! enumeration in the widest suite; both graphs are checked by the
    # exact relative-order route in every run.
    hub2 = hub_witness_edges(2)
    hub2_marg = mst_marginals(6, hub2)
    hub2_joint = mst_pair_probability(6, hub2, 0, 1)
    check((hub2_marg[0], hub2_marg[1], hub2_joint) ==
          (Fraction(1, 2), Fraction(1, 2), Fraction(1186, 4725)),
          "six-vertex hub witness: exact probabilities differ")
    check(hub2_joint / (hub2_marg[0] * hub2_marg[1]) == Fraction(4744, 4725),
          "six-vertex hub witness: ratio is not 4744/4725")
    if full:
        hub2_perm_marg, hub2_perm_pair = perm_mst_probabilities(6, hub2)
        check(hub2_perm_marg == hub2_marg,
              "six-vertex hub witness: permutation/pattern marginals disagree")
        check(hub2_perm_pair[(0, 1)] == hub2_joint,
              "six-vertex hub witness: permutation/pattern joints disagree")

    hub3 = hub_witness_edges(3)
    hub3_marg = mst_marginals(7, hub3)
    hub3_joint = mst_pair_probability(7, hub3, 0, 1)
    check((hub3_marg[0], hub3_marg[1], hub3_joint) ==
          (Fraction(1123, 2730), Fraction(1123, 2730),
           Fraction(71479, 420420)),
          "seven-vertex hub witness: exact probabilities differ")
    hub3_ratio = hub3_joint / (hub3_marg[0] * hub3_marg[1])
    check(hub3_ratio == Fraction(13938405, 13872419),
          "seven-vertex hub witness: ratio differs")
    check(hub3_ratio > Fraction(78100, 77841),
          "seven-vertex hub witness does not beat the bundle maximum")

    # A symmetry-compressed relative-order calculation reaches the finite
    # range stated in Proposition 4.4.  Its s=2,3 values are checked above by
    # the unspecialized route.
    hub_expected = {
        1: (Fraction(2, 3), Fraction(4, 9)),
        2: (Fraction(1, 2), Fraction(1186, 4725)),
        3: (Fraction(1123, 2730), Fraction(71479, 420420)),
        4: (Fraction(544933, 1531530), Fraction(24528499, 192972780)),
        5: (Fraction(568103, 1790712), Fraction(1017591682, 10082827755)),
        6: (Fraction(10731429, 37182145), Fraction(1676413313149, 20098436658300)),
        7: (Fraction(2606373997, 9786090600), Fraction(106303586434949, 1498769133661800)),
        8: (Fraction(320245982159, 1289317436550), Fraction(2561006549304523, 41571122812619400)),
        9: (Fraction(4951519969327, 21202108956600), Fraction(197400895109230339, 3629186192210178600)),
    }
    hub_ratios = []
    for s, expected in hub_expected.items():
        p_a, p_b, p_ab, ratio = hub_family_finite(s)
        check(p_a == p_b and (p_a, p_ab) == expected,
              f"hub family s={s}: finite-range exact values differ")
        hub_ratios.append(ratio)
    check(max(range(1, 10), key=lambda s: hub_ratios[s - 1]) == 3,
          "hub family: finite-range maximum is not uniquely at s=3")
    check(all(ratio < 1 for ratio in hub_ratios[6:]),
          "hub family: ratio is not below one for s=7,8,9")

    report("PASS block 1: MST measure self-tests -- p_0 = 2/n, sum_e p_e = n-1, "
           "two direct enumerations agree on K_4 and the five-vertex witness; "
           "LPS and stronger hub-witness values exact")


def _connected_simple(nv: int, edges, skip: int | None = None) -> bool:
    adj = [[] for _ in range(nv)]
    for k, (u, v) in enumerate(edges):
        if k == skip:
            continue
        adj[u].append(v)
        adj[v].append(u)
    seen = {0}
    stack = [0]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return len(seen) == nv


def _canonical_simple_code(nv: int, edges) -> int:
    edge_set = {tuple(sorted(edge)) for edge in edges}
    best = None
    for perm in permutations(range(nv)):
        code = 0
        bit = 0
        for i in range(nv):
            for j in range(i + 1, nv):
                if tuple(sorted((perm[i], perm[j]))) in edge_set:
                    code |= 1 << bit
                bit += 1
        if best is None or code < best:
            best = code
    assert best is not None
    return best


def block1_simple_minimality_census() -> None:
    """Every connected simple graph with at most seven edges is p-NC.

    A minimal counterexample may be assumed bridgeless: bridges are always in
    the MST, and deleting them separates independent MST problems.  A connected
    bridgeless graph has minimum degree at least two, hence nv <= m.  We census
    all such labelled graphs and evaluate one representative of every
    isomorphism class exactly.
    """
    representatives = {}
    labelled = 0
    for nv in range(3, 8):
        possible = list(combinations(range(nv), 2))
        for m in range(nv, 8):
            if m > len(possible):
                continue
            for edges in combinations(possible, m):
                degree = [0] * nv
                for u, v in edges:
                    degree[u] += 1
                    degree[v] += 1
                if min(degree) < 2 or not _connected_simple(nv, edges):
                    continue
                if any(not _connected_simple(nv, edges, k) for k in range(m)):
                    continue
                labelled += 1
                key = (nv, m, _canonical_simple_code(nv, edges))
                representatives.setdefault(key, edges)

    checked_pairs = 0
    for (nv, m, _), edges in representatives.items():
        edges = list(edges)
        marg = mst_marginals(nv, edges)
        for i, j in combinations(range(m), 2):
            joint = mst_pair_probability(nv, edges, i, j)
            check(joint <= marg[i] * marg[j],
                  f"simple minimality census: p-NC fails at {(nv, tuple(edges), i, j)}")
            checked_pairs += 1
    check(labelled == 1528,
          f"simple minimality census: expected 1528 labelled graphs, got {labelled}")
    check(len(representatives) == 17,
          "simple minimality census: expected 17 isomorphism representatives")
    check(checked_pairs == 272,
          f"simple minimality census: expected 272 marked pairs, got {checked_pairs}")
    report("PASS block 1: every connected simple graph with at most seven edges "
           "is p-NC (1528 bridgeless labelled graphs, 17 isomorphism classes, "
           "272 representative marked pairs)")


# ---------------------------------------------------------------------------
# block 2: the bridge from the MST measure to the coalescent
# ---------------------------------------------------------------------------

def _pair_probs_from_measure(n: int):
    """(p_1, p_2) for K_n, computed from the MST measure by order patterns."""
    edges = complete_graph_edges(n)
    index = {e: i for i, e in enumerate(edges)}
    p1 = mst_pair_probability(n, edges, index[(0, 1)], index[(0, 2)])
    p2 = None
    if n >= 4:
        p2 = mst_pair_probability(n, edges, index[(0, 1)], index[(2, 3)])
    return p1, p2


def block2_bridge_to_coalescent(full: bool) -> None:
    measured = {}
    for n in ((4, 5, 6) if full else (4, 5)):
        p1, p2 = _pair_probs_from_measure(n)
        measured[n] = (p1, p2)
        m = coalescent_moments(n)
        check(p1 == m["p1"], f"K_{n}: measure p_1 != coalescent p_1")
        check(p2 == m["p2"], f"K_{n}: measure p_2 != coalescent p_2")
        # E[Phi] read off the measure, against 8(n-1) - 4E[H].
        ephi = p1 * n * (n - 1) * (n - 2)
        check(ephi == 8 * (n - 1) - 4 * m["EH"], f"K_{n}: E[Phi] != 8(n-1) - 4E[H]")
        # E[deg^2] from the measure, against Proposition D1.
        ed2 = ephi / n + Fraction(2 * (n - 1), n)
        check(ed2 == Fraction(10 * (n - 1), n) - 4 * m["EL"],
              f"K_{n}: Proposition D1 fails against the measure")

    # Known exact values, so a silent change of convention is caught.
    check(measured[4] == (Fraction(17, 90), Fraction(11, 45)),
          "K_4: (p_1, p_2) != (17/90, 11/45)")
    check(measured[5] == (Fraction(919, 7560), Fraction(593, 3780)),
          "K_5: (p_1, p_2) != (919/7560, 593/3780)")

    sizes = "K_4, K_5 and K_6" if full else "K_4 and K_5"
    report("PASS block 2: the accepted-merger coalescent reproduces the MST "
           f"measure on {sizes} (p_1, p_2, E[Phi], Proposition D1)")


# ---------------------------------------------------------------------------
# block 3: the pathwise lemmas of section 5
# ---------------------------------------------------------------------------

def block3_pathwise(shape_max: int) -> None:
    def is_comb(node) -> bool:
        if node[1] is None:
            return True
        _, left, right = node
        if left[1] is None and right[1] is None:
            return True
        if left[1] is None:
            return is_comb(right)
        if right[1] is None:
            return is_comb(left)
        return False

    counts = []
    for n in range(1, shape_max + 1):
        shapes = hierarchy_shapes(n)
        counts.append(len(shapes))
        for shape in shapes:
            h, j = hierarchy_HJ(shape)
            check(h - j == n - Fraction(1, n), f"pathwise identity H - J = n - 1/n fails at n={n}")
            if n >= 2:
                check(j >= Fraction(n - 1, n), f"pathwise J >= (n-1)/n fails at n={n}")
                harmonic_bound = harmonic_number(n) - 1
                check(j >= harmonic_bound, f"pathwise J >= H_n-1 fails at n={n}")
                check((j == harmonic_bound) == is_comb(shape),
                      f"equality in J >= H_n-1 is not exactly the comb case at n={n}")
            if n >= 3:
                check(j > Fraction(n - 1, n), f"strict J > (n-1)/n fails at n={n}")
            check(8 * (n - 1) - 4 * h <= Fraction(4 * (n - 1) * (n - 2), n),
                  f"pathwise E[Phi | history] upper bound fails at n={n}")
    report(f"PASS block 3: pathwise H - J = n - 1/n and sharp J >= H_n-1 on every "
           f"merger-history shape, n <= {shape_max}; shape counts = {counts}")

    # (F4): the disjoint-pair bound is NOT pathwise.  Balanced hierarchy, n=4.
    balanced = None
    for shape in hierarchy_shapes(4):
        if shape[1][0] == 2 and shape[2][0] == 2:
            balanced = shape
    check(balanced is not None, "(F4): balanced n=4 hierarchy not found")
    if balanced is not None:
        h, j = hierarchy_HJ(balanced)
        check(j == Fraction(5, 4), "(F4): balanced n=4 hierarchy has J != 5/4")
        check(j > Fraction(9, 8), "(F4): balanced n=4 J does not exceed 9/8")
        check(8 * 3 - 4 * h == 4, "(F4): balanced n=4 conditional E[Phi] != 4")
        check(Fraction(4) < Fraction(3 * 3 * 2, 4),
              "(F4): the disjoint threshold 9/2 was not exceeded")
    report("PASS block 3 (F4): the disjoint-pair bound is not pathwise -- "
           "balanced n=4 history has J = 5/4 > 9/8 and E[Phi|history] = 4 < 9/2")


# ---------------------------------------------------------------------------
# block 4: the partition inequality and its equality case
# ---------------------------------------------------------------------------

def block4_partition(part_max: int) -> None:
    total = 0
    tight = 0
    for n in range(2, part_max + 1):
        for a in partitions(n):
            if len(a) < 2:
                continue
            lhs, rhs = partition_lhs_rhs(a)
            check(lhs <= rhs, f"one-step partition inequality fails at {a}")
            equal = lhs == rhs
            predicted = (len(a) == 2) or (len(set(a)) == 1)
            check(equal == predicted,
                  f"partition-inequality equality characterisation fails at {a}")
            tight += equal
            total += 1
    report(f"PASS block 4: one-step partition inequality and its equality case "
           f"on {total} partitions, 2 <= n <= {part_max} ({tight} equalities)")

    # (F5) the merger law matters: uniform block-pair choice breaks the bound.
    #
    # All three quantities are E[1/(a_i+a_j)] under a weighting on pairs, so
    # they must be compared in the SAME normalisation.  partition_lhs_rhs
    # returns the UNNORMALISED sums (11/6 and 15/8 at (2,1,1)); dividing by
    # sum_{i<j} a_i a_j turns them into the multiplicative-weighted mean 11/30
    # and the bound k/(2n) = 3/8.  The uniform-weighted mean is the plain
    # average over the k pairs, 7/18.  Comparing the raw sums against the
    # normalised constants is what this check previously did, and it failed.
    a = (2, 1, 1)
    n_a, k_a = sum(a), len(a)
    prod_sum = sum(x * y for x, y in combinations(a, 2))
    lhs, rhs = partition_lhs_rhs(a)
    multiplicative = Fraction(lhs, 1) / prod_sum
    bound = Fraction(rhs, 1) / prod_sum
    check(bound == Fraction(k_a, 2 * n_a),
          "(F5): normalised bound is not k/(2n) = 3/8")
    check(multiplicative == Fraction(11, 30) and multiplicative <= bound,
          "(F5): multiplicative value at (2,1,1) is not 11/30 <= 3/8")
    uniform = sum(Fraction(1, x + y) for x, y in combinations(a, 2)) / k_a
    check(uniform == Fraction(7, 18) and uniform > bound,
          "(F5): uniform block-pair value at (2,1,1) is not 7/18 > 3/8")
    report("PASS block 4 (F5): uniform block-pair choice gives 7/18 > 3/8 at "
           "(2,1,1), against the multiplicative value 11/30")


# ---------------------------------------------------------------------------
# block 5: E[L_n] by an independent route, and against Gamarnik's table
# ---------------------------------------------------------------------------

def block5_expected_length(poly_max: int) -> None:
    for n in range(2, poly_max + 1):
        m = coalescent_moments(n)
        check(m["EL"] == expected_mst_length_poly(n),
              f"E[L_{n}]: coalescent route != connectivity-polynomial route")
        check(m["EH"] == n - Fraction(1, n) + m["EJ"],
              f"H - J = n - 1/n in expectation fails at n={n}")
    check(coalescent_moments(2)["EL"] == 1, "E[L_2] != 1")
    check(coalescent_moments(3)["EL"] == Fraction(7, 6), "E[L_3] != 7/6")
    check(coalescent_moments(4)["EL"] == Fraction(73, 60), "E[L_4] != 73/60")
    report(f"PASS block 5: E[L_n] agrees between the coalescent and the "
           f"connectivity-polynomial route for 2 <= n <= {poly_max}; "
           f"E[H] = n - 1/n + E[J] is recomputed from merger increments")

    path = os.path.join(ROOT, "data", "gamarnik-2005-exp1-table.txt")
    compared = 0
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            n_str, val_str = line.split()
            n = int(n_str)
            if n > poly_max:
                continue
            published = Fraction(val_str)
            exact = coalescent_moments(n)["EL"]
            check(abs(exact - published) <= Fraction(1, 20000),
                  f"E[L_{n}] disagrees with Gamarnik's published table")
            compared += 1
    check(compared >= 10, "Gamarnik comparison covered too few values")
    report(f"PASS block 5: exact E[L_n] matches Gamarnik (SODA 2005, Table 1) to "
           f"every printed decimal, for {compared} values of n")


# ---------------------------------------------------------------------------
# block 6: the committed exact table, and Theorems B1/B2 on it
# ---------------------------------------------------------------------------

TABLE_FIELDS = [
    "n", "p0", "p1", "p2", "p0_squared", "E_deg_squared", "E_L_n",
    "E_L_n_upper_bound", "E_L_n_lower_bound", "E_L_n_harmonic_lower_bound",
    "p1_over_p0sq_float",
    "p2_over_p0sq_float",
]


def block6_table(recompute_max: int) -> None:
    path = os.path.join(ROOT, "data", "kn-exact-table.csv")
    rows = 0
    ns = []
    p1_ratios = []
    p2_ratios = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != TABLE_FIELDS:
            check(False, f"table schema differs: {reader.fieldnames!r}")
            report("PASS block 6: committed-table schema and rows checked")
            return
        for row in reader:
            n = int(row["n"])
            ns.append(n)
            p0 = Fraction(row["p0"])
            p1 = Fraction(row["p1"])
            p2 = Fraction(row["p2"]) if row["p2"] else None
            ed2 = Fraction(row["E_deg_squared"])
            el = Fraction(row["E_L_n"])
            hi = Fraction(row["E_L_n_upper_bound"])
            lo = Fraction(row["E_L_n_lower_bound"])
            harmonic_lo = Fraction(row["E_L_n_harmonic_lower_bound"])

            check(p0 == Fraction(2, n), f"table n={n}: p_0 != 2/n")
            check(p0 * p0 == Fraction(row["p0_squared"]), f"table n={n}: p_0^2")
            check(p1 < p0 * p0, f"Theorem B1 fails on the table at n={n}")
            p1_ratios.append(p1 / (p0 * p0))
            if p2 is not None:
                check(p2 < p0 * p0, f"Theorem B2 fails on the table at n={n}")
                p2_ratios.append(p2 / (p0 * p0))
            check((n == 3) == (row["p2"] == ""),
                  f"table n={n}: p2 must be blank exactly at n=3")
            check(ed2 == Fraction(10 * (n - 1), n) - 4 * el,
                  f"Proposition D1 fails on the table at n={n}")
            check(hi == Fraction((n - 1) * (5 * n + 6), 4 * n * n),
                  f"table n={n}: upper bound formula")
            check(lo == Fraction((n - 1) * (n + 2), n * n),
                  f"table n={n}: lower bound formula")
            check(harmonic_lo == 1 + (harmonic_number(n) - 1) / n - Fraction(1, n * n),
                  f"table n={n}: harmonic lower bound formula")
            check(lo <= harmonic_lo <= el,
                  f"table n={n}: strengthened harmonic lower bound fails")
            check(lo <= el <= hi, f"Corollary D2 window fails at n={n}")
            check((p1 <= p0 * p0) == (el >= lo),
                  f"Corollary D2 (adjacent direction) fails at n={n}")
            check(
                p1 / (p0 * p0)
                == (ed2 - 2 + Fraction(2, n))
                / (4 * (1 - Fraction(3, n) + Fraction(2, n * n))),
                f"pair-ratio identity for p1 fails at n={n}",
            )
            if p2 is not None:
                check((p2 <= p0 * p0) == (el <= hi),
                      f"Corollary D2 (disjoint direction) fails at n={n}")
                check(
                    p2 / (p0 * p0)
                    == Fraction(n * n * ((n - 1) - ed2),
                                (n - 1) * (n - 2) * (n - 3)),
                    f"pair-ratio identity for p2 fails at n={n}",
                )
            check(abs(Fraction(row["p1_over_p0sq_float"]) - p1 / (p0 * p0))
                  <= Fraction(1, 2 * 10 ** 10), f"table n={n}: p1 ratio decimal")
            if p2 is not None:
                check(abs(Fraction(row["p2_over_p0sq_float"]) - p2 / (p0 * p0))
                      <= Fraction(1, 2 * 10 ** 10), f"table n={n}: p2 ratio decimal")
            if n <= recompute_max:
                m = coalescent_moments(n)
                check(m["p1"] == p1 and m["p2"] == p2 and m["EL"] == el
                      and m["ED2"] == ed2,
                      f"table n={n}: committed row differs from recomputation")
            rows += 1
    check(ns == list(range(3, 31)),
          f"table domain/order must be exactly n=3,...,30, got {ns}")
    check(rows == 28, f"table must have exactly 28 rows, got {rows}")
    check(all(a < b for a, b in zip(p1_ratios, p1_ratios[1:])),
          "p1/p0^2 is not strictly increasing through n=30")
    check(all(a < b for a, b in zip(p2_ratios, p2_ratios[1:])),
          "p2/p0^2 is not strictly increasing through n=30")
    check(coalescent_moments(7)["EL"] < coalescent_moments(8)["EL"],
          "E[L_8] is not greater than E[L_7]")
    check(coalescent_moments(9)["EL"] < coalescent_moments(8)["EL"],
          "E[L_8] is not greater than E[L_9]")
    report(f"PASS block 6: the committed table of {rows} rows is internally "
           f"exact, satisfies Theorems B1 and B2 strictly and Corollary D2; "
           f"rows with n <= {recompute_max} were recomputed from scratch")


def block6_manuscript_sync() -> None:
    """Tie the manuscript's displayed finite constants and Table 1 to code."""
    path = os.path.join(ROOT, "paper", "main.tex")
    with open(path) as fh:
        tex = fh.read()

    computed_literals = {
        "simple ratio": str(Fraction(1450, 1449)),
        "bundle maximum": str(bundle_family(4, 4)[3]),
        "three-hub ratio": str(hub_family_finite(3)[3]),
        "Theorem C crossing": "first exceeding\n$8$ at $t=84$",
        "degree-weight identity": r"\frac{10(n-1)}n-4\,\E[L_n]",
        "harmonic bound": r"H_n-1",
        "public data location": r"https://github.com/agupta/random-mst-correlations",
    }
    for label, literal in computed_literals.items():
        check(literal in tex, f"manuscript source drift: {label}")

    match = re.search(
        r"\\begin\{tabular\}\{rllcll\}(.*?)\\end\{tabular\}", tex, re.S
    )
    check(match is not None, "manuscript source drift: Table 1 tabular not found")
    if match is None:
        report("PASS block 6: manuscript constants and Table 1 tied to exact sources")
        return

    rows = {}
    for line in match.group(1).splitlines():
        stripped = line.strip()
        if not re.match(r"^\$\d+\$\s*&", stripped):
            continue
        cells = [cell.strip().strip("$") for cell in stripped.removesuffix(r"\\").split("&")]
        rows[int(cells[0])] = cells[1:]

    shown = list(range(3, 15)) + [20, 30]
    check(sorted(rows) == shown, f"manuscript Table 1 domain differs: {sorted(rows)}")
    for n in shown:
        m = coalescent_moments(n)
        p0sq = Fraction(4, n * n)
        harmonic_lo = 1 + (harmonic_number(n) - 1) / n - Fraction(1, n * n)
        hi = Fraction((n - 1) * (5 * n + 6), 4 * n * n)
        expected = [
            f"{float(m['p1'] / p0sq):.10f}",
            "---" if m["p2"] is None else f"{float(m['p2'] / p0sq):.10f}",
            f"{float(m['EL']):.4f}",
            f"{float(harmonic_lo):.4f}",
            str(hi),
        ]
        check(rows.get(n) == expected,
              f"manuscript Table 1 row n={n} differs: {rows.get(n)!r}")
    report("PASS block 6: manuscript constants and Table 1 tied to exact sources")


# ---------------------------------------------------------------------------
# block 7: closing algebra of sections 5 and 6
# ---------------------------------------------------------------------------

def block7_algebra(algebra_max: int) -> None:
    for n in range(3, algebra_max + 1):
        check(8 * (n - 1) - 4 * (n + Fraction(n - 2, n))
              == Fraction(4 * (n - 1) * (n - 2), n), f"B1 algebra at n={n}")
        check(sum(Fraction(k, 2 * n) for k in range(2, n + 1))
              == Fraction((n - 1) * (n + 2), 4 * n), f"budget sum at n={n}")
        hbound = n - Fraction(1, n) + Fraction((n - 1) * (n + 2), 4 * n)
        check(hbound == Fraction((n - 1) * (5 * n + 6), 4 * n), f"H budget at n={n}")
        check(8 * (n - 1) - 4 * hbound == Fraction(3 * (n - 1) * (n - 2), n),
              f"B2 algebra at n={n}")
        el_lo = Fraction((n - 1) * (n + 2), n * n)
        el_hi = Fraction((n - 1) * (5 * n + 6), 4 * n * n)
        check(Fraction(10 * (n - 1), n) - 4 * el_lo
              == 6 - Fraction(14, n) + Fraction(8, n * n),
              f"Corollary 2.4 adjacent threshold at n={n}")
        check(Fraction(10 * (n - 1), n) - 4 * el_hi
              == 5 - Fraction(11, n) + Fraction(6, n * n),
              f"Corollary 2.4 disjoint threshold at n={n}")
    report(f"PASS block 7: the closing rational algebra of sections 5 and 6, and "
           f"the Tang-Zhang Corollary 2.4 dictionary rederived from Proposition "
           f"D1, for 3 <= n <= {algebra_max}")


# ---------------------------------------------------------------------------
# block 8: the negative results of section 7
# ---------------------------------------------------------------------------

BUNDLE_MAX_RATIO = Fraction(78100, 77841)


def _padd(a, b):
    out = [0] * max(len(a), len(b))
    for i, value in enumerate(a):
        out[i] += value
    for i, value in enumerate(b):
        out[i] += value
    while len(out) > 1 and out[-1] == 0:
        out.pop()
    return out


def _pscale(a, scalar):
    return [scalar * value for value in a]


def _pmul(a, b):
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += x * y
    return out


def _ppow(a, exponent):
    out = [1]
    base = a
    while exponent:
        if exponent & 1:
            out = _pmul(out, base)
        base = _pmul(base, base)
        exponent //= 2
    return out


def _pshift_variable(a, constant):
    """Coefficients of a(u + constant), ascending in powers of u."""
    out = [0]
    base = [constant, 1]
    for exponent, coefficient in enumerate(a):
        out = _padd(out, _pscale(_ppow(base, exponent), coefficient))
    return out


def _peval(a, value):
    return sum(coefficient * value ** exponent
               for exponent, coefficient in enumerate(a))


@lru_cache(maxsize=1)
def _bundle_rho_parts():
    """The seven univariate polynomials printed in Proposition 4.2."""
    m = [0, 1]
    D = _pmul(_pmul(_padd(m, [2]), _padd(m, [3])), _padd(m, [4]))
    K = _pmul(_ppow(_padd(m, [2]), 2),
              _pmul(_padd(m, [3]), _padd(m, [4])))
    L = _padd(D, [24])
    C = _padd(_pscale(D, 6), [32])
    N0 = _pmul(
        K,
        _padd(
            _padd(_pscale(_ppow(m, 3), 6), _pscale(_ppow(m, 2), 78)),
            _padd(_pscale(m, 356), [544]),
        ),
    )
    N1 = _pmul(K, _padd(_padd(_ppow(m, 2), _pscale(m, 7)), [16]))
    D0 = _padd(
        _padd(_pscale(_pmul(C, _ppow(m, 2)), 4), _pmul(_pmul(L, C), m)),
        _ppow(C, 2),
    )
    D1 = _padd(_padd(_ppow(L, 2), _pscale(_pmul(L, m), 4)),
               _pscale(C, -8))
    return N0, N1, D0, D1


def _bundle_rho_ratio(r, s):
    m, rho = r + s, r * s
    N0, N1, D0, D1 = _bundle_rho_parts()
    return Fraction(
        _peval(N0, m) + _peval(N1, m) * rho,
        _peval(D0, m) + _peval(D1, m) * rho + 16 * rho * rho,
    )


def _check_bundle_certificate() -> None:
    N0, N1, D0, D1 = _bundle_rho_parts()
    m_minus_one = [-1, 1]
    q_unbalanced = _padd(
        _padd(
            _padd(_pmul(N1, D0), _pscale(_pmul(N0, D1), -1)),
            _pscale(_pmul(N0, m_minus_one), -32),
        ),
        _pscale(_pmul(N1, _ppow(m_minus_one, 2)), -16),
    )
    negative_q_shifted = _pshift_variable(_pscale(q_unbalanced, -1), 11)
    check(
        negative_q_shifted == [
            930527357760, 3158322926304, 1989328507888, 609067397832,
            112783678672, 13814498848, 1164466368, 68281008, 2747600,
            72576, 1136, 8,
        ],
        "bundle certificate: shifted -Q_m(m-1) coefficients differ",
    )

    numerator = _padd(N0, _pmul(N1, m_minus_one))
    denominator = _padd(
        _padd(D0, _pmul(D1, m_minus_one)),
        _pscale(_ppow(m_minus_one, 2), 16),
    )
    gap = _padd(_pscale(denominator, 78100),
                _pscale(numerator, -77841))
    check(
        _pshift_variable(gap, 11) == [
            34642293840, 28712778516, 9667374224, 1607247273,
            144551244, 7147790, 181300, 1813,
        ],
        "bundle certificate: shifted endpoint-gap coefficients differ",
    )

    q_values = []
    for m_value in range(2, 11):
        rho = m_value * m_value // 4
        q_values.append(
            _peval(_pmul(N1, D0), m_value)
            - _peval(_pmul(N0, D1), m_value)
            - 32 * _peval(N0, m_value) * rho
            - 16 * _peval(N1, m_value) * rho * rho
        )
    check(q_values == [
        623232000, 3726475200, 15482880000, 50485284864,
        133754572800, 297101597760, 541023436800,
        751512453120, 464792380416,
    ], "bundle certificate: small-m Q values differ")


def block8_bundles(sum_max: int) -> None:
    _check_bundle_certificate()
    argmax = None
    best = Fraction(0)
    pairs = 0
    for r in range(1, sum_max):
        for s in range(1, sum_max - r + 1):
            ratio = bundle_family(r, s)[3]
            check(ratio == _bundle_rho_ratio(r, s),
                  f"bundle rho formula differs at (r,s)=({r},{s})")
            check(ratio <= BUNDLE_MAX_RATIO,
                  f"bundle bound 78100/77841 fails at (r,s)=({r},{s})")
            pairs += 1
            if ratio > best:
                best, argmax = ratio, (r, s)
    check(best == BUNDLE_MAX_RATIO and argmax == (4, 4),
          "bundle bound: the maximum is not 78100/77841 at (4,4)")
    pe, pf, pef, lps_ratio = bundle_family(3, 3)
    check(pe == pf == Fraction(331, 1260) and pef == Fraction(109, 1575),
          "bundle family does not reproduce the LPS probabilities at (3,3)")
    check(lps_ratio == Fraction(109872, 109561),
          "bundle family does not reproduce the LPS ratio at (3,3)")

    # The (4,4) formula against a direct relative-order enumeration of the
    # underlying 12-edge multigraph.
    bundle44_edges = (
        [(0, 1)] * 4 + [(2, 3)] * 4
        + [(0, 2), (0, 3), (1, 2), (1, 3)]
    )
    bundle44_marg = mst_marginals(4, bundle44_edges)
    bundle44_joint = mst_pair_probability(4, bundle44_edges, 0, 4)
    check((bundle44_marg[0], bundle44_marg[4], bundle44_joint)
          == (Fraction(93, 440), Fraction(93, 440), Fraction(71, 1584)),
          "bundle (4,4): direct graph probabilities differ")
    check(bundle44_joint / (bundle44_marg[0] * bundle44_marg[4])
          == BUNDLE_MAX_RATIO,
          "bundle (4,4): direct graph ratio differs")

    for t in range(1, sum_max):
        ratio = bundle_family(t, t)[3]
        den = (2 * t ** 3 + 17 * t * t + 34 * t + 22) ** 2
        closed = Fraction(
            (t + 1) ** 2 * (t + 4) * (2 * t + 3) * (2 * t * t + 19 * t + 34), den)
        check(ratio == closed, f"symmetric bundle closed form fails at t={t}")
        check(ratio - 1 == Fraction(20 * t ** 3 + 9 * t * t - 78 * t - 76, den),
              f"symmetric bundle excess closed form fails at t={t}")
    report(f"PASS block 8: the bundle ratio is <= 78100/77841 on all {pairs} "
           f"pairs with r+s <= {sum_max}, attained only at (4,4); LPS is "
           f"recovered at (3,3); printed certificates and direct (4,4) "
           f"enumeration verified")


def block8_nonidentical(t_max: int, mono_max: int) -> None:
    # Exact all-integer monotonicity certificate.  The sign of R(t+1)-R(t)
    # is the sign of N(t+1)D(t)-N(t)D(t+1); after t=u+1 every coefficient is
    # positive.
    tpoly = [0, 1]
    numerator = _pmul(
        _pmul(
            _pmul(_padd(_pscale(tpoly, 5), [6]), _padd(tpoly, [1])),
            _ppow(_padd(tpoly, [4]), 2),
        ),
        _padd(_pscale(tpoly, 2), [3]),
    )
    denominator = _ppow(
        _padd(_padd(_pscale(_ppow(tpoly, 2), 5), _pscale(tpoly, 12)), [8]),
        2,
    )
    monotonicity_gap = _padd(
        _pmul(_pshift_variable(numerator, 1), denominator),
        _pscale(_pmul(numerator, _pshift_variable(denominator, 1)), -1),
    )
    check(_pshift_variable(monotonicity_gap, 1) == [
        124000, 669300, 1289264, 1262269, 712831,
        242311, 48935, 5400, 250,
    ], "Theorem C: all-integer monotonicity certificate differs")

    for t in range(1, t_max + 1):
        pa, pb, pab, ratio = maxlaw_family(t)
        pa2, pb2, pab2, _ = maxlaw_family_by_integration(t)
        check((pa, pb, pab) == (pa2, pb2, pab2),
              f"Theorem C: closed form != direct Stieltjes integration at t={t}")
        closed = Fraction(
            2 * (5 * t + 6) * (t + 1) * (t + 4) ** 2 * (2 * t + 3),
            9 * (5 * t * t + 12 * t + 8) ** 2)
        check(ratio == closed, f"Theorem C: ratio closed form fails at t={t}")
    check(maxlaw_family(1)[0] == Fraction(1, 2), "Theorem C: p_A(1) != 1/2")
    check(maxlaw_family(1)[2] == Fraction(11, 45), "Theorem C: p_AB(1) != 11/45")
    for t, want in ((1, Fraction(44, 45)), (2, Fraction(168, 169)),
                    (3, Fraction(8232, 7921)), (4, Fraction(2860, 2601))):
        check(maxlaw_family(t)[3] == want, f"Theorem C: R({t}) != {want}")

    prev = None
    for t in range(1, mono_max + 1):
        cur = maxlaw_family(t)[3]
        if prev is not None:
            check(cur > prev, f"Theorem C: R is not increasing at t={t}")
        prev = cur
    check(maxlaw_family(83)[3] <= 8 < maxlaw_family(84)[3],
          "Theorem C: first crossing of 8 is not t=84")
    check(maxlaw_family(3000)[3] > 267, "Theorem C: R(3000) <= 267")
    big = maxlaw_family(10 ** 6)[3] / (10 ** 6)
    check(abs(big - Fraction(4, 45)) < Fraction(1, 1000),
          "Theorem C: R(t)/t does not approach 4/45")
    report(f"PASS block 8: Theorem C verified two ways for t <= {t_max}, "
           f"increasing for t <= {mono_max}, first R(t)>8 at t=84, "
           f"R(3000) > 267, "
           f"R(t)/t -> 4/45")


def block8_subdivision_regression() -> None:
    """Check Lemma 4.1 and the attenuation identity against the i.i.d.
    subdivided graph itself, rather than against another specialization of the
    four-terminal formulas."""
    for t in (1, 2, 3):
        next_vertex = 4

        def path_edges(start, end):
            nonlocal next_vertex
            vertices = [start]
            for _ in range(t - 1):
                vertices.append(next_vertex)
                next_vertex += 1
            vertices.append(end)
            return list(zip(vertices, vertices[1:]))

        path_a = path_edges(0, 1)
        path_b = path_edges(2, 3)
        edges = path_a + path_b + [(0, 2), (0, 3), (1, 2), (1, 3)]
        index_b = len(path_a)
        marg = mst_marginals(next_vertex, edges)
        joint = mst_pair_probability(next_vertex, edges, 0, index_b)
        macro_p, _, macro_joint, _ = maxlaw_family(t)
        alpha = 1 - (1 - macro_p) / t
        predicted_joint = alpha * alpha + (macro_joint - macro_p * macro_p) / (t * t)
        check(marg[0] == marg[index_b] == alpha,
              f"subdivision t={t}: fixed-edge marginal identity fails")
        check(joint == predicted_joint,
              f"subdivision t={t}: attenuation covariance identity fails")
    report("PASS block 8: Theorem C and attenuation checked against the exact "
           "i.i.d. subdivision graphs for t=1,2,3")


# ---------------------------------------------------------------------------
# block 9: Theorem A on an exhaustive small census
# ---------------------------------------------------------------------------

def _connected(nv: int, edges) -> bool:
    adj = {v: [] for v in range(nv)}
    for u, v in edges:
        if u != v:
            adj[u].append(v)
            adj[v].append(u)
    seen = {0}
    stack = [0]
    while stack:
        x = stack.pop()
        for y in adj[x]:
            if y not in seen:
                seen.add(y)
                stack.append(y)
    return len(seen) == nv


def block9_theorem_a(edge_max: int) -> None:
    """Theorem A, instance by instance, on every connected labelled multigraph
    on 3 or 4 vertices with at most `edge_max` edges, loops included."""
    graphs = 0
    pairs = 0
    worst = Fraction(0)
    worst_graph = None
    for nv in (3, 4):
        types = [(u, v) for u in range(nv) for v in range(u, nv)]
        for m in range(nv - 1, edge_max + 1):
            for choice in combinations_with_replacement(range(len(types)), m):
                edges = [types[i] for i in choice]
                if not _connected(nv, edges):
                    continue
                marg = mst_marginals(nv, edges)
                check(sum(marg) == nv - 1,
                      f"census: sum_e p_e != n-1 on {(nv, tuple(edges))}")
                for i, j in combinations(range(m), 2):
                    pij = mst_pair_probability(nv, edges, i, j)
                    check(pij <= 8 * marg[i] * marg[j],
                          f"Theorem A fails on {(nv, tuple(edges))} at {(i, j)}")
                    if marg[i] > 0 and marg[j] > 0:
                        ratio = pij / (marg[i] * marg[j])
                        if ratio > worst:
                            worst, worst_graph = ratio, (nv, tuple(edges), i, j)
                    pairs += 1
                graphs += 1
    check(graphs > 0 and pairs > 100, "census: refusing a vacuous universe")
    report(f"PASS block 9: Theorem A verified exactly on {graphs} connected "
           f"labelled multigraphs ({pairs} marked pairs, at most {edge_max} "
           f"edges); largest ratio in this universe {worst} "
           f"= {float(worst):.7f} at {worst_graph}")


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="wider ranges; minutes rather than about one minute")
    args = ap.parse_args()
    full = args.full

    shape_max = 12 if full else 11
    part_max = 32 if full else 26
    poly_max = 30 if full else 16
    recompute_max = 30
    algebra_max = 400 if full else 200
    bundle_sum_max = 120 if full else 60
    t_max = 12 if full else 8
    mono_max = 300 if full else 120
    census_edges = 6 if full else 5

    report(f"check_paper_claims.py  (full={full})")
    block1_measure_self_tests(full)
    if full:
        block1_simple_minimality_census()
    block2_bridge_to_coalescent(full)
    block3_pathwise(shape_max)
    block4_partition(part_max)
    block5_expected_length(poly_max)
    block6_table(recompute_max)
    block6_manuscript_sync()
    block7_algebra(algebra_max)
    block8_bundles(bundle_sum_max)
    block8_nonidentical(t_max, mono_max)
    block8_subdivision_regression()
    block9_theorem_a(census_edges)

    report("")
    if FAILURES:
        report(f"FAILED: {len(FAILURES)} of {PASSES + len(FAILURES)} checks")
        for line in FAILURES[:40]:
            report(f"  FAIL  {line}")
        return 1
    report(f"ALL CHECKS PASSED ({PASSES} exact checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
