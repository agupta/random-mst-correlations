#!/usr/bin/env python3
"""Independent exact checks for the proposed universal MST factor 8.

This file deliberately does not import finder code.  It uses only Python's standard
library and Fraction arithmetic.  The exhaustive census is finite evidence, not a
replacement for the proof in Section 3 of paper/main.tex.
"""

import sys
from fractions import Fraction as F
from itertools import combinations, combinations_with_replacement, permutations
from math import factorial

if sys.flags.optimize:
    raise RuntimeError("this checker requires asserts; do not run Python with -O")


class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, x, y):
        x, y = self.find(x), self.find(y)
        if x == y:
            return False
        self.p[y] = x
        return True


def is_connected(n, edges):
    if n <= 1:
        return True
    d = DSU(n)
    for u, v in edges:
        if u != v:
            d.union(u, v)
    return len({d.find(v) for v in range(n)}) == 1


def kruskal_mask(n, edges, order):
    d = DSU(n)
    mask = 0
    for i in order:
        u, v = edges[i]
        if u != v and d.union(u, v):
            mask |= 1 << i
    return mask


def all_mst_counts(n, edges):
    """Counts all marginals and pairs over every strict edge ordering."""
    m = len(edges)
    total = factorial(m)
    marg = [0] * m
    joint = [[0] * m for _ in range(m)]
    tree_sizes = set()
    for order in permutations(range(m)):
        mask = kruskal_mask(n, edges, order)
        chosen = [i for i in range(m) if mask >> i & 1]
        tree_sizes.add(len(chosen))
        for i in chosen:
            marg[i] += 1
        for i, j in combinations(chosen, 2):
            joint[i][j] += 1
            joint[j][i] += 1
    return total, marg, joint, tree_sizes


def marked_mst_counts(n, edges, e, f):
    """Memory-light exact enumeration for one marked pair."""
    total = factorial(len(edges))
    ce = cf = cef = 0
    for order in permutations(range(len(edges))):
        mask = kruskal_mask(n, edges, order)
        ie, jf = (mask >> e) & 1, (mask >> f) & 1
        ce += ie
        cf += jf
        cef += ie & jf
    return total, ce, cf, cef


def threshold_fixed(n, weighted_edges, x, y):
    """The deleted-graph bottleneck threshold for one fixed environment."""
    if x == y:
        return F(0)
    d = DSU(n)
    for weight, u, v in sorted(weighted_edges):
        if u != v:
            d.union(u, v)
        if d.find(x) == d.find(y):
            return weight
    return F(1)


def threshold_order_moments(n, edges, x, y):
    """E[T], E[T^2] via connection rank and uniform order-statistic moments."""
    m = len(edges)
    if x == y:
        return F(0), F(0)
    if not terminals_connected(n, edges, x, y):
        return F(1), F(1)
    s1 = s2 = F(0)
    for order in permutations(range(m)):
        d = DSU(n)
        rank = None
        for k, i in enumerate(order, 1):
            u, v = edges[i]
            if u != v:
                d.union(u, v)
            if d.find(x) == d.find(y):
                rank = k
                break
        assert rank is not None
        s1 += F(rank, m + 1)
        s2 += F(rank * (rank + 1), (m + 1) * (m + 2))
    den = factorial(m)
    return s1 / den, s2 / den


def terminals_connected(n, edges, x, y):
    if x == y:
        return True
    d = DSU(n)
    for u, v in edges:
        if u != v:
            d.union(u, v)
    return d.find(x) == d.find(y)


def percolation_moments(n, edges, x, y):
    """Independently integrate the exact disconnection polynomial.

    The beta integrals used are
      integral t^k(1-t)^(m-k) dt = k!(m-k)!/(m+1)!,
      integral 2t*t^k(1-t)^(m-k) dt = 2(k+1)!(m-k)!/(m+2)!.
    The convention T=1 for topologically disconnected terminals is included.
    """
    if x == y:
        return F(0), F(0)
    m = len(edges)
    s1 = s2 = F(0)
    for mask in range(1 << m):
        d = DSU(n)
        k = 0
        for i, (u, v) in enumerate(edges):
            if mask >> i & 1:
                k += 1
                if u != v:
                    d.union(u, v)
        if d.find(x) != d.find(y):
            s1 += F(factorial(k) * factorial(m - k), factorial(m + 1))
            s2 += F(2 * factorial(k + 1) * factorial(m - k), factorial(m + 2))
    return s1, s2


def q_value(n, edges, x, y, p):
    """q(p)=P(T>p), with the proof's q(1)=0 endpoint convention."""
    if p >= 1:
        return F(0)
    if x == y:
        return F(0)
    m = len(edges)
    ans = F(0)
    for mask in range(1 << m):
        d = DSU(n)
        k = 0
        for i, (u, v) in enumerate(edges):
            if mask >> i & 1:
                k += 1
                if u != v:
                    d.union(u, v)
        if d.find(x) != d.find(y):
            ans += p**k * (1 - p) ** (m - k)
    return ans


def conditional_areas(n, edges, e, f, h_order, profile):
    """Exactly integrate over the marked weights U,V for fixed H weights.

    Rectangles between consecutive H weights have constant order type.  A diagonal
    rectangle splits into two equal-area triangles U<V and V<U.
    """
    h_ids = [i for i in range(len(edges)) if i not in (e, f)]
    h = len(h_ids)
    if profile == 0:
        ranked_weights = [F(r, h + 1) for r in range(1, h + 1)]
    elif profile == 1:
        ranked_weights = [F(r * r, (h + 1) ** 2) for r in range(1, h + 1)]
    else:
        raise AssertionError("unknown profile")
    weights = {h_ids[idx]: ranked_weights[r] for r, idx in enumerate(h_order)}
    cuts = [F(0)] + ranked_weights + [F(1)]
    a = b = c = F(0)

    def add_region(u, v, area):
        nonlocal a, b, c
        full_weights = dict(weights)
        full_weights[e], full_weights[f] = u, v
        order = tuple(sorted(range(len(edges)), key=full_weights.__getitem__))
        mask = kruskal_mask(n, edges, order)
        ie, jf = (mask >> e) & 1, (mask >> f) & 1
        a += area * ie
        b += area * jf
        c += area * ie * jf

    for i in range(h + 1):
        li, ri = cuts[i], cuts[i + 1]
        wi = ri - li
        for j in range(h + 1):
            lj, rj = cuts[j], cuts[j + 1]
            wj = rj - lj
            if i != j:
                add_region((li + ri) / 2, (lj + rj) / 2, wi * wj)
            else:
                add_region(li + wi / 3, li + 2 * wi / 3, wi * wi / 2)
                add_region(li + 2 * wi / 3, li + wi / 3, wi * wi / 2)

    weighted_h = [(weights[i], *edges[i]) for i in h_ids]
    A = threshold_fixed(n, weighted_h, *edges[e])
    B = threshold_fixed(n, weighted_h, *edges[f])
    return a, b, c, A, B


def graph_universe():
    """A reproducible exhaustive labeled multigraph universe.

    It includes loops, arbitrary parallel classes, all connected labeled examples on
    2 vertices through 5 edges, on 3 vertices through 4 edges, loopless 3-vertex
    examples through 6 edges, loopless 4-vertex examples through 5 edges, and K4.
    """
    seen = set()
    specs = [
        (2, True, range(1, 6)),
        (3, True, range(2, 5)),
        (3, False, range(5, 7)),
        (4, False, range(3, 6)),
    ]
    for n, loops, sizes in specs:
        types = [(u, v) for u in range(n) for v in range(u if loops else u + 1, n)]
        for m in sizes:
            for choice in combinations_with_replacement(range(len(types)), m):
                edges = tuple(types[i] for i in choice)
                key = (n, edges)
                if key not in seen and is_connected(n, edges):
                    seen.add(key)
                    yield key
    k4 = (4, tuple((u, v) for u in range(4) for v in range(u + 1, 4)))
    if k4 not in seen:
        yield k4


def main():
    graphs = list(graph_universe())
    assert graphs, "refusing vacuous empty census"
    graph_checks = pair_checks = conditional_checks = deletion_checks = 0
    moment_checks = q_checks = 0
    strict_conditional_seen = False
    threshold_cases = {}

    for n, edges in graphs:
        total, marg, joint, sizes = all_mst_counts(n, edges)
        assert sizes == {n - 1}
        assert sum(marg) == (n - 1) * total
        graph_checks += 2
        m = len(edges)
        for e, f in combinations(range(m), 2):
            # Exact C=8 after clearing the two probability denominators.
            assert joint[e][f] * total <= 8 * marg[e] * marg[f]
            pair_checks += 1
            h_ids = [i for i in range(m) if i not in (e, f)]
            h_edges = tuple(edges[i] for i in h_ids)
            for terminals in (edges[e], edges[f]):
                key = (n, h_edges, terminals)
                threshold_cases[key] = None
            for h_order in permutations(range(m - 2)):
                for profile in (0, 1):
                    a, b, c, A, B = conditional_areas(
                        n, edges, e, f, h_order, profile
                    )
                    assert c <= a * b
                    assert A / 2 <= A - A * A / 2 <= a <= A
                    assert B / 2 <= B - B * B / 2 <= b <= B
                    strict_conditional_seen |= c < a * b
                    conditional_checks += 1
                    deletion_checks += 4

    assert strict_conditional_seen

    q_grid = ((F(0), F(0)), (F(1, 5), F(1, 5)),
              (F(1, 5), F(2, 5)), (F(1, 4), F(1, 3)),
              (F(1, 3), F(1, 3)), (F(2, 5), F(1, 2)),
              (F(1, 2), F(1, 2)))
    for n, edges, terminals in threshold_cases:
        x, y = terminals
        om = threshold_order_moments(n, edges, x, y)
        pm = percolation_moments(n, edges, x, y)
        assert om == pm
        et, et2 = om
        assert et2 <= 2 * et * et
        moment_checks += 2
        for s, t in q_grid:
            assert s + t <= 1
            assert q_value(n, edges, x, y, s + t) <= (
                q_value(n, edges, x, y, s) * q_value(n, edges, x, y, t)
            )
            q_checks += 1

    # Primary LPS example: K4 with the disjoint 01 and 23 edges tripled.
    lps = ((0, 1),) * 3 + ((2, 3),) * 3 + (
        (0, 2), (0, 3), (1, 2), (1, 3)
    )
    total, ce, cf, cef = marked_mst_counts(4, lps, 0, 3)
    pe, pf, pef = F(ce, total), F(cf, total), F(cef, total)
    ratio = pef / (pe * pf)
    assert pe == pf == F(331, 1260)
    assert pef == F(109, 1575)
    assert ratio == F(109872, 109561)
    assert pef <= 8 * pe * pf

    # Hostile false-claim controls: the checker must reject these strengthenings.
    assert pef > pe * pf                         # false universal C=1
    par6 = ((0, 1),) * 6
    et, et2 = threshold_order_moments(2, par6, 0, 1)
    assert et == F(1, 7) and et2 == F(1, 28)
    assert et2 > F(3, 2) * et * et              # false moment factor 3/2
    a, b, c, A, B = conditional_areas(2, ((0, 1), (0, 1)), 0, 1, (), 0)
    assert (a, b, c, A, B) == (F(1, 2), F(1, 2), F(0), F(1), F(1))
    assert a < F(3, 4) * A                      # false stronger deletion lower bound
    a, b, c, A, B = conditional_areas(3, ((0, 1), (1, 2)), 0, 1, (), 0)
    assert (a, b, c) == (F(1), F(1), F(1))
    assert c == a * b                           # false strict conditional inequality
    total, marg, joint, sizes = all_mst_counts(2, ((0, 0), (0, 1)))
    assert marg == [0, total] and sizes == {1}  # loop and bridge branches
    assert percolation_moments(3, ((0, 1),), 0, 2) == (F(1), F(1))
    assert q_value(3, ((0, 1),), 0, 2, F(1)) == 0  # T=1 survival endpoint

    print("PASS: independent stdlib exact MST factor-8 audit")
    print(f"PASS: exhaustive universe = {len(graphs)} connected labeled multigraphs")
    print(f"PASS: graph/tree identities = {graph_checks}")
    print(f"PASS: exact C=8 marked-pair inequalities = {pair_checks}")
    print(f"PASS: exact conditional Harris-area instances = {conditional_checks}")
    print(f"PASS: pointwise deleted-threshold inequalities = {deletion_checks}")
    print(f"PASS: order-statistic/percolation moment identities and factor 2 = {moment_checks}")
    print(f"PASS: exact q(s+t)<=q(s)q(t) grid instances = {q_checks}")
    print(f"PASS: LPS p(e)=p(f)={pe}, p(e,f)={pef}, ratio={ratio}>1")
    print("PASS: hostile controls reject C=1, moment factor 3/2, deletion 3A/4, and strict Harris")
    print("PASS: explicit loop, bridge, parallel-edge, disconnected-H, and q(1)=0 controls")


if __name__ == "__main__":
    main()
