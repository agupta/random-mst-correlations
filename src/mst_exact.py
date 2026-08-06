"""Exact rational computations for random minimum spanning trees.

Everything here is exact: integers and `fractions.Fraction` only.  No floating
point is used in any decision path.  Standard library only.

The module provides four exact routes with explicit independence boundaries:

1. `perm_mst_probabilities`   -- brute force over all m! edge orderings.
2. `mst_marginals` / `mst_pair_probability`
                              -- exact summation over the 2^(m-1) resp.
                                 3^(m-2) relative-order patterns.
3. `coalescent_moments`       -- the accepted-merger (multiplicative
                                 coalescent) dynamic program on K_n.
4. `expected_mst_length_poly` -- E[L_n] via the Kruskal area identity and the
                                 exact G(n,p) connectivity polynomial; this
                                 route knows nothing about the coalescent.

Routes 1 and 2 are distinct enumerations of Kruskal's rule.  Route 3 implements
the coalescent proof model, while route 4 computes E[L_n] without that model.
The tests use the direct routes and route 4 as external checks on route 3.
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from itertools import permutations, combinations
from math import comb, factorial


# ---------------------------------------------------------------------------
# union-find
# ---------------------------------------------------------------------------

class DSU:
    __slots__ = ("p",)

    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        p = self.p
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, x: int, y: int) -> bool:
        a, b = self.find(x), self.find(y)
        if a == b:
            return False
        self.p[a] = b
        return True


def complete_graph_edges(n: int):
    """Edge list of K_n, vertices 0..n-1."""
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def _connected_after(nv: int, edges, mask_indices) -> DSU:
    d = DSU(nv)
    for k in mask_indices:
        u, v = edges[k]
        d.union(u, v)
    return d


# ---------------------------------------------------------------------------
# route 1: brute force over all orderings
# ---------------------------------------------------------------------------

def perm_mst_probabilities(nv: int, edges):
    """Exact marginal and pairwise MST probabilities by enumerating all m!
    orderings of the edges.  Returns (marginals, pairs) with Fraction values.

    The MST of a graph with distinct edge weights depends on the weights only
    through their order, so averaging over the m! equally likely orders is
    exact for i.i.d. atomless weights.
    """
    m = len(edges)
    total = factorial(m)
    marg = [0] * m
    pair = {(i, j): 0 for i in range(m) for j in range(i + 1, m)}
    for order in permutations(range(m)):
        d = DSU(nv)
        chosen = []
        for k in order:
            u, v = edges[k]
            if d.union(u, v):
                chosen.append(k)
        for k in chosen:
            marg[k] += 1
        for a, b in combinations(sorted(chosen), 2):
            pair[(a, b)] += 1
    marg = [Fraction(c, total) for c in marg]
    pair = {k: Fraction(c, total) for k, c in pair.items()}
    return marg, pair


# ---------------------------------------------------------------------------
# route 2: exact summation over relative-order patterns
# ---------------------------------------------------------------------------

def mst_marginals(nv: int, edges):
    """P(edge k in MST) for every k, exactly.

    Kruskal: edge k is accepted iff its endpoints lie in different components
    of the graph formed by the edges strictly lighter than k.  Conditioning on
    the *set* S of lighter edges, P(S is exactly that set) = 1 / (m C(m-1,|S|)).
    """
    m = len(edges)
    out = []
    for k in range(m):
        others = [i for i in range(m) if i != k]
        acc = Fraction(0)
        for mask in range(1 << (m - 1)):
            sub = [others[t] for t in range(m - 1) if mask >> t & 1]
            d = _connected_after(nv, edges, sub)
            u, v = edges[k]
            if d.find(u) != d.find(v):
                acc += Fraction(1, m * comb(m - 1, len(sub)))
        out.append(acc)
    return out


def mst_pair_probability(nv: int, edges, e: int, f: int) -> Fraction:
    """P(edges e and f are both in the MST), exactly.

    Split the other m-2 edges into the three blocks (below e, between e and f,
    above f) for each of the two orders of e and f.  A given ordered 3-block
    pattern has probability |S1|! |S2|! |S3|! / m!.
    """
    m = len(edges)
    others = [i for i in range(m) if i != e and i != f]
    r = len(others)
    total = Fraction(0)
    for lo, hi in ((e, f), (f, e)):
        ulo, vlo = edges[lo]
        uhi, vhi = edges[hi]
        for pattern in range(3 ** r):
            p = pattern
            s1, s2, s3 = [], [], []
            for t in range(r):
                block = p % 3
                p //= 3
                (s1 if block == 0 else s2 if block == 1 else s3).append(others[t])
            d1 = _connected_after(nv, edges, s1)
            if d1.find(ulo) == d1.find(vlo):
                continue                      # lo rejected
            d2 = _connected_after(nv, edges, s1 + s2 + [lo])
            if d2.find(uhi) == d2.find(vhi):
                continue                      # hi rejected
            total += Fraction(
                factorial(len(s1)) * factorial(len(s2)) * factorial(len(s3)),
                factorial(m),
            )
    return total


# ---------------------------------------------------------------------------
# symmetry-compressed relative-order enumeration for the simple hub family
# ---------------------------------------------------------------------------

def _hub_join(partition, subset):
    """Merge the terminal blocks touched by one hub's lighter incident edges."""
    subset = frozenset(subset)
    if len(subset) <= 1:
        return partition
    touched = [block for block in partition if block & subset]
    untouched = [block for block in partition if not block & subset]
    merged = frozenset().union(*touched)
    return tuple(sorted(untouched + [merged], key=min))


def _hub_connected(partition, x: int, y: int) -> bool:
    return any(x in block and y in block for block in partition)


@lru_cache(maxsize=None)
def hub_family_finite(s: int):
    """Exact probabilities for the simple ``s``-hub family.

    The four terminals are 0,1,2,3, the marked edges are 01 and 23, and each
    of ``s`` independent hub vertices is joined to all four terminals.  The
    calculation groups the 2^(4s) marginal patterns and 3^(4s) pair patterns
    by the two induced terminal partitions and their block sizes.  It is an
    exact symmetry-compressed version of route 2, intended for the explicitly
    finite range used in the manuscript.
    """
    if s < 1:
        raise ValueError("s >= 1 required")

    discrete = tuple(frozenset((i,)) for i in range(4))
    m = 4 * s + 2

    # Marginal of 01.  The other marked edge 23 is either below or above it.
    marginal_states = {(discrete, 0): 1}
    for _ in range(s):
        nxt = {}
        for (partition, below), count in marginal_states.items():
            for mask in range(16):
                subset = tuple(i for i in range(4) if mask >> i & 1)
                key = (_hub_join(partition, subset), below + len(subset))
                nxt[key] = nxt.get(key, 0) + count
        marginal_states = nxt

    p_a = Fraction(0)
    for (partition, below), count in marginal_states.items():
        for marked_below in (False, True):
            tested = _hub_join(partition, (2, 3)) if marked_below else partition
            if _hub_connected(tested, 0, 1):
                continue
            k = below + int(marked_below)
            p_a += count * Fraction(factorial(k) * factorial(m - 1 - k), factorial(m))

    # Joint probability, first on the region w(01)<w(23).  For each hub,
    # ``low`` is the subset of incident edges below the first marked weight and
    # ``middle`` the subset between the marked weights.
    choices = []
    for low_mask in range(16):
        for middle_mask in range(16):
            if low_mask & middle_mask:
                continue
            low = tuple(i for i in range(4) if low_mask >> i & 1)
            through_middle = tuple(
                i for i in range(4) if (low_mask | middle_mask) >> i & 1
            )
            choices.append((low, through_middle, len(low), middle_mask.bit_count()))

    pair_states = {(discrete, discrete, 0, 0): 1}
    for _ in range(s):
        nxt = {}
        for (low_part, middle_part, below, middle), count in pair_states.items():
            for low, through_middle, add_below, add_middle in choices:
                key = (
                    _hub_join(low_part, low),
                    _hub_join(middle_part, through_middle),
                    below + add_below,
                    middle + add_middle,
                )
                nxt[key] = nxt.get(key, 0) + count
        pair_states = nxt

    one_order = Fraction(0)
    for (low_part, middle_part, below, middle), count in pair_states.items():
        if _hub_connected(low_part, 0, 1):
            continue
        with_first = _hub_join(middle_part, (0, 1))
        if _hub_connected(with_first, 2, 3):
            continue
        above = m - 2 - below - middle
        one_order += count * Fraction(
            factorial(below) * factorial(middle) * factorial(above), factorial(m)
        )

    p_ab = 2 * one_order
    return p_a, p_a, p_ab, p_ab / (p_a * p_a)


# ---------------------------------------------------------------------------
# route 3: the accepted-merger coalescent on K_n
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def coalescent_moments(n: int):
    """Exact E[J], E[H], E[Phi], E[D^2], p1, p2 and E[L_n] for K_n.

    States are partitions of n into component sizes.  From a state with sizes
    a_1..a_k the next accepted edge joins blocks i<j with probability
    a_i a_j / Z, Z = sum_{i<j} a_i a_j.

    J  = sum over the n-1 mergers of 1/(parent size),
    H  = sum over the n-1 mergers of (1/a + 1/b) over child sizes,
    Phi = sum_x deg(x)(deg(x)-1) in the finished tree,
    L_n = total MST weight for i.i.d. Exp(1) edge weights.
    """
    if n < 2:
        raise ValueError("n >= 2 required")
    dist = {(1,) * n: Fraction(1)}
    EJ = Fraction(0)
    EH = Fraction(0)
    EL = Fraction(0)
    while True:
        nxt = {}
        progressed = False
        for state, prob in dist.items():
            k = len(state)
            if k == 1:
                continue
            progressed = True
            s2 = sum(a * a for a in state)
            Z = Fraction(n * n - s2, 2)
            dJ = Fraction(0)
            dH = Fraction(0)
            for i in range(k):
                for j in range(i + 1, k):
                    a, b = state[i], state[j]
                    dJ += Fraction(a * b, a + b)
                    # Accumulate H from its merger increment under the actual
                    # transition weights, not from the identity E[H] = n E[L].
                    dH += Fraction(a * b, a) + Fraction(a * b, b)
            EJ += prob * dJ / Z
            EH += prob * dH / Z
            EL += prob * Fraction(k - 1) / Z
            for i in range(k):
                for j in range(i + 1, k):
                    a, b = state[i], state[j]
                    rest = list(state[:i] + state[i + 1:j] + state[j + 1:])
                    rest.append(a + b)
                    ns = tuple(sorted(rest))
                    nxt[ns] = nxt.get(ns, Fraction(0)) + prob * Fraction(a * b) / Z
        if not progressed:
            break
        dist = nxt
    EPhi = 8 * (n - 1) - 4 * EH
    ED2 = EPhi / n + Fraction(2 * (n - 1), n)
    p1 = EPhi / (n * (n - 1) * (n - 2)) if n >= 3 else None
    p2 = (
        (Fraction((n - 1) * (n - 2)) - EPhi) * 4
        / (n * (n - 1) * (n - 2) * (n - 3))
        if n >= 4
        else None
    )
    return {
        "EJ": EJ, "EH": EH, "EPhi": EPhi, "ED2": ED2,
        "p1": p1, "p2": p2, "EL": EL,
    }


# ---------------------------------------------------------------------------
# route 4: E[L_n] from the Kruskal area identity, independent of route 3
# ---------------------------------------------------------------------------

def _polymul(a, b):
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                out[i + j] += x * y
    return out


def _shift(a, k):
    return [0] * k + list(a)


def _polyadd(a, b):
    n = max(len(a), len(b))
    return [(a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0)
            for i in range(n)]


@lru_cache(maxsize=None)
def _connectivity_poly(s: int):
    """Probability that G(s, p) is connected, as an integer polynomial in
    u = 1 - p.  D_1 = 1, D_s = 1 - sum_{a<s} C(s-1,a-1) D_a u^{a(s-a)}."""
    if s == 1:
        return (1,)
    acc = [0]
    for a in range(1, s):
        term = _polymul(_connectivity_poly(a), (comb(s - 1, a - 1),))
        acc = _polyadd(acc, _shift(term, a * (s - a)))
    res = _polyadd([1], [-c for c in acc])
    return tuple(res)


def expected_mst_length_poly(n: int) -> Fraction:
    """E[L_n] for i.i.d. Exp(1) weights, by Kruskal's area identity

        L_n = int_0^inf (kappa(t) - 1) dt,

    where kappa(t) is the number of components of the graph of edges of weight
    at most t, i.e. of G(n, p) with p = 1 - e^{-t}.  Substituting u = e^{-t}
    turns E[kappa(t)] - 1 into an integer polynomial in u with zero constant
    term, and dt = -du/u, so the integral is sum_j c_j / j.
    """
    if n < 2:
        raise ValueError("n >= 2 required")
    kap = [0]
    for s in range(1, n + 1):
        term = _polymul(_connectivity_poly(s), (comb(n, s),))
        kap = _polyadd(kap, _shift(term, s * (n - s)))
    kap = _polyadd(kap, [-1])           # kappa - 1
    assert kap[0] == 0, "constant term must vanish"
    return sum(Fraction(c, j) for j, c in enumerate(kap) if j >= 1 and c)


# ---------------------------------------------------------------------------
# rooted binary hierarchies: the pathwise identity and the pathwise J bound
# ---------------------------------------------------------------------------

def hierarchy_shapes(n: int):
    """All rooted binary hierarchies on n unlabelled leaves, as nested tuples
    of sizes; each node is (size, left, right) with leaves (1, None, None).

    Shapes are generated up to left/right swap, which is all the pathwise
    functionals H and J depend on."""
    @lru_cache(maxsize=None)
    def gen(m):
        if m == 1:
            return ((1, None, None),)
        out = []
        for a in range(1, m // 2 + 1):
            b = m - a
            for L in gen(a):
                for R in gen(b):
                    if a == b and L > R:
                        continue
                    out.append((m, L, R))
        return tuple(out)
    return gen(n)


def hierarchy_HJ(node):
    """(H, J) for a hierarchy: H sums 1/a + 1/b over child sizes at each
    merger, J sums 1/(a+b) over parent sizes."""
    if node[1] is None:
        return Fraction(0), Fraction(0)
    m, L, R = node
    hL, jL = hierarchy_HJ(L)
    hR, jR = hierarchy_HJ(R)
    a, b = L[0], R[0]
    return (hL + hR + Fraction(1, a) + Fraction(1, b), jL + jR + Fraction(1, m))


# ---------------------------------------------------------------------------
# the one-step partition inequality
# ---------------------------------------------------------------------------

def partitions(n: int, cap: int | None = None):
    """All partitions of n as non-increasing tuples."""
    if cap is None:
        cap = n
    if n == 0:
        yield ()
        return
    for first in range(min(n, cap), 0, -1):
        for rest in partitions(n - first, first):
            yield (first,) + rest


def partition_lhs_rhs(a):
    """Left and right sides of sum_{i<j} a_i a_j/(a_i+a_j) <= (k/2n) sum a_i a_j."""
    n = sum(a)
    k = len(a)
    lhs = sum(Fraction(x * y, x + y) for x, y in combinations(a, 2))
    Z = sum(Fraction(x * y) for x, y in combinations(a, 2))
    return lhs, Fraction(k, 2 * n) * Z


# ---------------------------------------------------------------------------
# the K_4 opposite-edge families
# ---------------------------------------------------------------------------

def bundle_family(r: int, s: int):
    """Exact P(e), P(f), P(e,f) for one marked copy in each of two opposite
    parallel bundles of sizes r and s on K_4, the other four edges uniform.

    Returns (p_e, p_f, p_ef, ratio).  (r,s) = (3,3) is the Lyons-Peres-Schramm
    configuration.
    """
    D = Fraction((r + s + 2) * (r + s + 3) * (r + s + 4))
    PA = r * (Fraction(r + 6, (r + 2) * (r + 4)) + 4 / D)
    PB = s * (Fraction(s + 6, (s + 2) * (s + 4)) + 4 / D)

    def I(x, y):
        return (Fraction(4, (y + 2) * (x + y + 3))
                - Fraction(3 * y + 10, (y + 2) * (y + 4) * (x + y + 4)))

    PAB = r * s * (I(r, s) + I(s, r))
    pe, pf, pef = PA / r, PB / s, PAB / (r * s)
    return pe, pf, pef, pef / (pe * pf)


def maxlaw_family(t: int):
    """Exact P(e), P(f), P(e,f) on K_4 where the two opposite edges 01 and 23
    carry the law max(U_1,...,U_t) (c.d.f. x^t) and the other four edges are
    uniform.  Returns (p_e, p_f, p_ef, ratio)."""
    A = Fraction(3 * (5 * t * t + 12 * t + 8),
                 (t + 1) * (t + 2) * (t + 4) * (2 * t + 3))
    J = Fraction(2 * (5 * t + 6), (t + 1) * (t + 2) ** 2 * (2 * t + 3))
    return A, A, J, J / (A * A)


def maxlaw_family_by_integration(t: int):
    """The same three probabilities recomputed from the four-terminal Stieltjes
    formulas, by exact polynomial integration.  Used as an independent check of
    `maxlaw_family`.

    With F = G = x^t for the two opposite edges and H = x for the four cross
    edges, all integrands are polynomials in x and the integrals are exact.
    """
    def integrate(coeffs):
        # coeffs[k] is the coefficient of x^k; returns int_0^1
        return sum(Fraction(c, k + 1) for k, c in enumerate(coeffs) if c)

    def mul(a, b):
        return _polymul(a, b)

    def add(*ps):
        out = [0]
        for p in ps:
            out = _polyadd(out, p)
        return out

    def scale(p, c):
        return [c * x for x in p]

    Hp = [0, 1]                       # H(x) = x
    Hbar = [1, -1]                    # 1 - H(x)
    Hbar2 = mul(Hbar, Hbar)
    Hbar4 = mul(Hbar2, Hbar2)
    one_minus_H2 = add([1], scale(mul(Hp, Hp), -1))          # 1 - H^2
    G = [0] * t + [1]                                        # G(x) = x^t
    dF = [0] * (t - 1) + [t] if t >= 1 else [0]              # dF/dx = t x^{t-1}

    # p_A = int [ (1-G)(1-H^2)^2 + G(2 Hbar^2 - Hbar^4) ] dF
    integrand = add(
        mul(add([1], scale(G, -1)), mul(one_minus_H2, one_minus_H2)),
        mul(G, add(scale(Hbar2, 2), scale(Hbar4, -1))),
    )
    pA = integrate(mul(integrand, dF))

    # p_AB = 2 * int_{x<y} Hbar(y)^2 [ 2(1-H(x)^2) - Hbar(y)^2 ] dF(x) dG(y)
    # by symmetry of F and G here.  Do the inner x-integral as a polynomial
    # in y, then integrate in y.
    dG = dF
    # inner(y) = int_0^y [2(1-H(x)^2)] dF(x)  and  int_0^y dF(x) = F(y)
    inner_a = _antiderivative(mul(scale(one_minus_H2, 2), dF))
    Fy = _antiderivative(dF)
    Hbar_y2 = Hbar2
    Hbar_y4 = Hbar4
    outer = add(mul(Hbar_y2, inner_a), scale(mul(Hbar_y4, Fy), -1))
    pAB = 2 * integrate(mul(outer, dG))
    return pA, pA, pAB, pAB / (pA * pA)


def _antiderivative(p):
    return [0] + [Fraction(c, k + 1) for k, c in enumerate(p)]
