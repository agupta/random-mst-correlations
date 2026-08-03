#!/usr/bin/env python3
"""Independent exact checks for the multiplicative-coalescent T2 proof.

Only the Python standard library is used.  Every calculation is integer or Fraction.
"""

from __future__ import annotations

import argparse
import sys
from fractions import Fraction
from functools import lru_cache

if sys.flags.optimize:
    raise RuntimeError("this checker requires asserts; do not run Python with -O")


def integer_partitions(n: int, lo: int = 1):
    """Nondecreasing positive integer tuples summing to n."""
    if n == 0:
        yield ()
        return
    for first in range(lo, n + 1):
        for tail in integer_partitions(n - first, first):
            yield (first,) + tail


def transition_quantities(pi: tuple[int, ...]):
    n, k = sum(pi), len(pi)
    assert k >= 2
    z = sum(pi[i] * pi[j] for i in range(k) for j in range(i + 1, k))
    a = sum(
        (Fraction(pi[i] * pi[j], pi[i] + pi[j])
         for i in range(k) for j in range(i + 1, k)),
        Fraction(),
    )
    return a, z, Fraction(k, 2 * n)


def merged(pi: tuple[int, ...], i: int, j: int):
    return tuple(sorted(
        [pi[t] for t in range(len(pi)) if t != i and t != j]
        + [pi[i] + pi[j]]
    ))


@lru_cache(None)
def expected_j(pi: tuple[int, ...]):
    """Exact expected remaining sum of 1/(a+b), with MC weights ab/Z."""
    if len(pi) == 1:
        return Fraction()
    k = len(pi)
    z = sum(pi[i] * pi[j] for i in range(k) for j in range(i + 1, k))
    ans = Fraction()
    mass = Fraction()
    for i in range(k):
        for j in range(i + 1, k):
            probability = Fraction(pi[i] * pi[j], z)
            mass += probability
            ans += probability * (Fraction(1, pi[i] + pi[j])
                                  + expected_j(merged(pi, i, j)))
    assert mass == 1
    return ans


LEAF = ()


def shape_key(shape):
    return repr(shape)


@lru_cache(None)
def shapes(n: int):
    """Canonical unordered full binary hierarchy shapes with n leaves."""
    if n == 1:
        return (LEAF,)
    out = set()
    for a in range(1, n // 2 + 1):
        b = n - a
        for left in shapes(a):
            for right in shapes(b):
                if a == b and shape_key(left) > shape_key(right):
                    continue
                pair = tuple(sorted((left, right), key=shape_key))
                out.add(pair)
    return tuple(sorted(out, key=shape_key))


@lru_cache(None)
def shape_stats(shape):
    """Return (leaf count, H, J, sum of local factorial increments)."""
    if shape == LEAF:
        return 1, Fraction(), Fraction(), Fraction()
    left, right = shape
    a, hl, jl, fl = shape_stats(left)
    b, hr, jr, fr = shape_stats(right)
    h = hl + hr + Fraction(1, a) + Fraction(1, b)
    j = jl + jr + Fraction(1, a + b)
    factorial = fl + fr + 8 - Fraction(4, a) - Fraction(4, b)
    return a + b, h, j, factorial


def check_partitions(max_n: int):
    if max_n < 2:
        raise ValueError("partition range is empty; max_n must be at least 2")
    count = 0
    tight = 0
    for n in range(2, max_n + 1):
        for pi in integer_partitions(n):
            if len(pi) < 2:
                continue
            a, z, rhs = transition_quantities(pi)
            lhs = a / z
            assert lhs <= rhs, (pi, lhs, rhs)
            # Independently check the cleared identity/lower bound used in the proof.
            q = sum(x * x for x in pi)
            diff = sum(
                Fraction((pi[i] - pi[j]) ** 2, pi[i] + pi[j])
                for i in range(len(pi)) for j in range(i + 1, len(pi))
            )
            assert 4 * a == (len(pi) - 1) * n - diff
            assert diff >= Fraction(len(pi) * q - n * n, n)
            tight += lhs == rhs
            count += 1
    print(f"PASS transition inequality on {count} integer partitions, 2<=n<={max_n}"
          f" ({tight} equalities)")


def check_shapes(max_n: int):
    if max_n < 2:
        raise ValueError("shape range is empty; max_n must be at least 2")
    count = 0
    per_n = []
    for n in range(1, max_n + 1):
        sn = shapes(n)
        per_n.append(len(sn))
        for shape in sn:
            size, h, j, factorial = shape_stats(shape)
            assert size == n
            assert h - j == n - Fraction(1, n)
            assert factorial == 8 * (n - 1) - 4 * h
            count += 1
    print(f"PASS pathwise H-J and factorial charge on {count} unordered shapes,"
          f" n<={max_n}; counts={per_n}")

    # The forbidden arbitrary-hierarchy strengthening really fails.
    balanced4 = ((LEAF, LEAF), (LEAF, LEAF))
    _, _, _, fac = shape_stats(balanced4)
    mu = Fraction(3, 2)
    conditional_variance = fac / 4 + mu - mu * mu
    target_variance = Fraction(3, 8)
    assert conditional_variance == Fraction(1, 4) < target_variance
    print("PASS obstruction: balanced n=4 hierarchy has Var(D)=1/4 < 3/8")


def check_actual_mc(max_n: int):
    if max_n < 2:
        raise ValueError("MC range is empty; max_n must be at least 2")
    for n in range(2, max_n + 1):
        ej = expected_j((1,) * n)
        j_bound = Fraction((n - 1) * (n + 2), 4 * n)
        eh = n - Fraction(1, n) + ej
        h_bound = Fraction((n - 1) * (5 * n + 6), 4 * n)
        ef = 8 * (n - 1) - 4 * eh
        factorial_target = Fraction(3 * (n - 1) * (n - 2), n)
        assert ej <= j_bound
        assert eh <= h_bound
        assert ef >= factorial_target
    # A recognizable cold-check value, independent of the tagged-degree DP.
    assert expected_j((1, 1, 1, 1)) == Fraction(67, 60)
    print(f"PASS exact actual-MC recursion and factorial target for 2<=n<={max_n};"
          " E[J_4]=67/60")


def check_final_algebra(max_n: int):
    if max_n < 4:
        raise ValueError("final-algebra range is empty; max_n must be at least 4")
    for n in range(4, max_n + 1):
        sum_k = sum(range(2, n + 1))
        assert Fraction(sum_k, 2 * n) == Fraction((n - 1) * (n + 2), 4 * n)
        h_bound = n - Fraction(1, n) + Fraction(sum_k, 2 * n)
        assert h_bound == Fraction((n - 1) * (5 * n + 6), 4 * n)
        factorial = 8 * (n - 1) - 4 * h_bound
        assert factorial == Fraction(3 * (n - 1) * (n - 2), n)
        root_fac = factorial / n
        root_second = root_fac + Fraction(2 * (n - 1), n)
        assert root_second == 5 - Fraction(11, n) + Fraction(6, n * n)
        p1 = root_fac / ((n - 1) * (n - 2))
        assert p1 == Fraction(3, n * n)
    print(f"PASS final exact algebra and Tang--Zhang moment threshold for 4<=n<={max_n}")


def check_hostile_mutant():
    pi = (1, 1, 2)
    a, z, rhs = transition_quantities(pi)
    true_lhs = a / z
    uniform_lhs = sum(
        (Fraction(1, pi[i] + pi[j])
         for i in range(3) for j in range(i + 1, 3)),
        Fraction(),
    ) / 3
    assert true_lhs == Fraction(11, 30) <= rhs == Fraction(3, 8)
    assert uniform_lhs == Fraction(7, 18) > rhs
    print("PASS hostile mutant: uniform block-pair gives 7/18 > 3/8 at (1,1,2),"
          " while multiplicative gives 11/30")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition-max", type=int, default=32)
    parser.add_argument("--shape-max", type=int, default=12)
    parser.add_argument("--mc-max", type=int, default=18)
    parser.add_argument("--algebra-max", type=int, default=300)
    args = parser.parse_args()
    check_partitions(args.partition_max)
    check_shapes(args.shape_max)
    check_actual_mc(args.mc_max)
    check_final_algebra(args.algebra_max)
    check_hostile_mutant()
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
