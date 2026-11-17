"""Error-dependence diagnostics for panels of LLM judges, verifiers and models.

A *panel* is a set of k checkers scored on the same n items against ground
truth.  The only input is, for each item and checker, whether that checker was
wrong.  Everything here is pure standard library (Python 3.10+).

The central question: if k checkers agree, is that k independent pieces of
evidence, or one piece repeated k times?
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

__all__ = ["Panel", "PairStat", "MajorityStat", "Interval", "Report"]


# ----------------------------------------------------------------------------
# small numeric helpers
# ----------------------------------------------------------------------------

def _phi(n11: int, na: int, nb: int, n: int) -> Optional[float]:
    """Phi (Pearson) correlation of two binary columns from their counts."""
    denom = na * (n - na) * nb * (n - nb)
    if denom == 0:
        return None
    return (n * n11 - na * nb) / math.sqrt(denom)


def _log_choose(n: int, r: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(r + 1) - math.lgamma(n - r + 1)


def _fisher_greater(n11: int, na: int, nb: int, n: int) -> float:
    """One-sided Fisher exact p: P(X >= n11), X ~ Hypergeometric(n, na, nb)."""
    lo = max(0, na + nb - n)
    hi = min(na, nb)
    total = 0.0
    denom = _log_choose(n, nb)
    for x in range(max(n11, lo), hi + 1):
        total += math.exp(_log_choose(na, x) + _log_choose(n - na, nb - x) - denom)
    return min(1.0, total)


def _poisson_binomial_tail(ps: Sequence[float], at_least: int) -> float:
    """P(number of successes >= at_least) for independent Bernoulli(ps)."""
    dist = [1.0]
    for p in ps:
        nxt = [0.0] * (len(dist) + 1)
        for s, mass in enumerate(dist):
            nxt[s] += mass * (1.0 - p)
            nxt[s + 1] += mass * p
        dist = nxt
    return sum(dist[at_least:])


def _pack(bits: Sequence[int]) -> int:
    out = 0
    for i, b in enumerate(bits):
        if b:
            out |= 1 << i
    return out


# ----------------------------------------------------------------------------
# result containers
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class PairStat:
    a: str
    b: str
    both_wrong: int
    expected_both_wrong: float
    lift: Optional[float]
    phi: Optional[float]
    p_b_wrong_given_a_wrong: Optional[float]
    p_b_wrong_given_a_right: Optional[float]
    fisher_p: float


@dataclass(frozen=True)
class MajorityStat:
    threshold: int
    observed: float
    expected_if_independent: float
    all_wrong_observed: float
    all_wrong_expected_if_independent: float

    @property
    def ratio(self) -> Optional[float]:
        if self.expected_if_independent == 0:
            return None
        return self.observed / self.expected_if_independent


@dataclass(frozen=True)
class Interval:
    estimate: Optional[float]
    lo: Optional[float]
    hi: Optional[float]


@dataclass
class Report:
    n_items: int
    names: list
    error_rates: list
    mean_phi: Interval
    permutation_p: Optional[float]
    n_perm: int
    n_eff_participation: Interval
    n_eff_kish: Interval
    majority: MajorityStat
    pairs: list
    redundancy: dict
    warnings: list = field(default_factory=list)
    cluster_bootstrap: bool = False

    def to_dict(self) -> dict:
        def iv(x: Interval):
            return {"estimate": x.estimate, "lo": x.lo, "hi": x.hi}

        return {
            "n_items": self.n_items,
            "n_checkers": len(self.names),
            "checkers": [
                {"name": n, "error_rate": r} for n, r in zip(self.names, self.error_rates)
            ],
            "mean_pairwise_phi": iv(self.mean_phi),
            "dependence_permutation_p": self.permutation_p,
            "n_permutations": self.n_perm,
            "effective_checkers_participation": iv(self.n_eff_participation),
            "effective_checkers_kish": iv(self.n_eff_kish),
            "majority_wrong": {
                "threshold_wrong_votes": self.majority.threshold,
                "observed": self.majority.observed,
                "expected_if_independent": self.majority.expected_if_independent,
                "ratio": self.majority.ratio,
                "all_wrong_observed": self.majority.all_wrong_observed,
                "all_wrong_expected_if_independent": self.majority.all_wrong_expected_if_independent,
            },
            "pairs": [p.__dict__ for p in self.pairs],
            "redundancy_mean_phi_with_others": self.redundancy,
            "warnings": self.warnings,
            "bootstrap_unit": "cluster" if self.cluster_bootstrap else "item",
        }

    def __str__(self) -> str:  # human-readable report
        def f(x, d=3):
            return "n/a" if x is None else f"{x:.{d}f}"

        def pct(x):
            return f"{100 * x:.1f}%"

        k = len(self.names)
        out = [f"errordep report: {k} checkers, {self.n_items} items", ""]
        out.append("Error rates")
        for n, r in zip(self.names, self.error_rates):
            out.append(f"  {n:<24} {pct(r)}")
        out.append("")
        out.append(
            f"Dependence between checkers' errors: mean pairwise phi = {f(self.mean_phi.estimate)} "
            f"(95% CI {f(self.mean_phi.lo)} to {f(self.mean_phi.hi)})"
        )
        out.append(
            f"  permutation test for dependence (one-sided, {self.n_perm} permutations): "
            f"p = {f(self.permutation_p, 4)}"
        )
        out.append("")
        out.append("Effective number of independent checkers (out of %d)" % k)
        for label, iv in (
            ("Kish (independent votes)", self.n_eff_kish),
            ("participation (dimensions)", self.n_eff_participation),
        ):
            out.append(f"  {label:<26} {f(iv.estimate, 2)} (95% CI {f(iv.lo, 2)} to {f(iv.hi, 2)})")
        out.append("")
        m = self.majority
        out.append(f"Panel wrong when >= {m.threshold} of {k} checkers are wrong")
        out.append(
            f"  observed {pct(m.observed)} vs {pct(m.expected_if_independent)} expected "
            f"if errors were independent"
            + (f" ({m.ratio:.1f}x)" if m.ratio is not None else "")
        )
        out.append(
            f"  all {k} wrong: observed {pct(m.all_wrong_observed)} vs "
            f"{pct(m.all_wrong_expected_if_independent)} expected if independent"
        )
        out.append("")
        out.append("Most dependent pairs (by lift = observed / expected both-wrong)")
        ranked = sorted(
            (p for p in self.pairs if p.lift is not None), key=lambda p: -p.lift
        )[:5]
        for p in ranked:
            out.append(
                f"  {p.a} + {p.b}: both wrong {p.both_wrong} (expected {p.expected_both_wrong:.1f}), "
                f"lift {p.lift:.2f}, phi {f(p.phi, 2)}, "
                f"P(b wrong | a wrong) {f(p.p_b_wrong_given_a_wrong, 2)} vs "
                f"{f(p.p_b_wrong_given_a_right, 2)} if a right"
            )
        out.append("")
        out.append("Redundancy (mean phi with the other checkers)")
        for n, v in self.redundancy.items():
            out.append(f"  {n:<24} {f(v, 2)}")
        if self.warnings:
            out.append("")
            out.append("Warnings")
            out.extend(f"  - {w}" for w in self.warnings)
        out.append("")
        out.append(
            "Note: dependence here includes shared item difficulty. It shows the checkers "
            "do not supply independent evidence, not why."
        )
        return "\n".join(out)


# ----------------------------------------------------------------------------
# Panel
# ----------------------------------------------------------------------------

class Panel:
    """Errors of k checkers on n items.

    ``wrong[i][j]`` is truthy when checker ``j`` was wrong on item ``i``.
    ``clusters`` optionally gives a group id per item (for example a shared
    source document); the bootstrap then resamples whole groups.
    """

    def __init__(
        self,
        wrong: Sequence[Sequence[int]],
        names: Optional[Sequence[str]] = None,
        clusters: Optional[Sequence] = None,
    ):
        rows = [tuple(1 if v else 0 for v in r) for r in wrong]
        if not rows:
            raise ValueError("panel needs at least one item")
        k = len(rows[0])
        if k < 2:
            raise ValueError("panel needs at least two checkers")
        if any(len(r) != k for r in rows):
            raise ValueError("every item must have a value for every checker")
        if names is None:
            names = [f"c{j}" for j in range(k)]
        names = list(names)
        if len(names) != k or len(set(names)) != k:
            raise ValueError("names must be unique, one per checker")
        if clusters is not None and len(clusters) != len(rows):
            raise ValueError("clusters must have one entry per item")
        self._rows = rows
        self.names = names
        self.clusters = list(clusters) if clusters is not None else None
        self.n = len(rows)
        self.k = k
        self._cols = [_pack([r[j] for r in rows]) for j in range(k)]
        self._cnt = [c.bit_count() for c in self._cols]

    # -- constructors --------------------------------------------------------

    @classmethod
    def from_verdicts(cls, verdicts, gold, names=None, clusters=None) -> "Panel":
        """``verdicts[i][j]`` is checker j's answer on item i; ``gold[i]`` is the truth."""
        if len(verdicts) != len(gold):
            raise ValueError("verdicts and gold must have the same number of items")
        wrong = [[v != g for v in row] for row, g in zip(verdicts, gold)]
        return cls(wrong, names, clusters)

    @classmethod
    def from_correct(cls, correct, names=None, clusters=None) -> "Panel":
        return cls([[not v for v in row] for row in correct], names, clusters)

    # -- basic statistics ----------------------------------------------------

    @property
    def error_rates(self) -> list:
        return [c / self.n for c in self._cnt]

    def _pair_n11(self, i: int, j: int) -> int:
        return (self._cols[i] & self._cols[j]).bit_count()

    def pairwise(self) -> list:
        out = []
        n = self.n
        for i in range(self.k):
            for j in range(i + 1, self.k):
                na, nb = self._cnt[i], self._cnt[j]
                n11 = self._pair_n11(i, j)
                exp = na * nb / n
                out.append(
                    PairStat(
                        a=self.names[i],
                        b=self.names[j],
                        both_wrong=n11,
                        expected_both_wrong=exp,
                        lift=(n11 / exp) if exp > 0 else None,
                        phi=_phi(n11, na, nb, n),
                        p_b_wrong_given_a_wrong=(n11 / na) if na else None,
                        p_b_wrong_given_a_right=((nb - n11) / (n - na)) if n - na else None,
                        fisher_p=_fisher_greater(n11, na, nb, n),
                    )
                )
        return out

    def _phis(self) -> list:
        n = self.n
        vals = []
        for i in range(self.k):
            for j in range(i + 1, self.k):
                p = _phi(self._pair_n11(i, j), self._cnt[i], self._cnt[j], n)
                if p is not None:
                    vals.append(p)
        return vals

    def mean_phi(self) -> Optional[float]:
        vals = self._phis()
        return sum(vals) / len(vals) if vals else None

    # -- effective number of independent checkers -----------------------------

    def effective_checkers(self, method: str = "kish") -> Optional[float]:
        """Effective number of independent checkers.

        ``kish`` (default): m / (1 + (m-1) * mean phi), with mean phi floored at 0.
        This is the design-effect answer to "how many independent votes is this
        panel worth", the right reading for averaging or voting.

        ``participation``: m^2 / sum(R_ij^2) for the m x m correlation matrix R of the
        non-constant checkers' error indicators (the participation ratio of R's
        eigenvalues).  It counts independent *dimensions* of error and squares the
        correlations, so it is always at least the Kish value and understates
        dependence for voting questions.  Identical checkers give 1, independent
        ones give m under both.
        """
        n = self.n
        keep = [i for i in range(self.k) if 0 < self._cnt[i] < n]
        m = len(keep)
        if m < 2:
            return None
        if method == "kish":
            vals = []
            for a in range(m):
                for b in range(a + 1, m):
                    i, j = keep[a], keep[b]
                    vals.append(_phi(self._pair_n11(i, j), self._cnt[i], self._cnt[j], n))
            rho = max(0.0, sum(vals) / len(vals))
            return m / (1 + (m - 1) * rho)
        if method == "participation":
            sq = float(m)  # diagonal
            for a in range(m):
                for b in range(a + 1, m):
                    i, j = keep[a], keep[b]
                    p = _phi(self._pair_n11(i, j), self._cnt[i], self._cnt[j], n)
                    sq += 2 * p * p
            return m * m / sq
        raise ValueError("method must be 'participation' or 'kish'")

    # -- majority vote -------------------------------------------------------

    def majority_vote(self, threshold: Optional[int] = None) -> MajorityStat:
        """How often the panel is wrong versus what independence would predict.

        The panel counts as wrong when at least ``threshold`` checkers are wrong
        (default: a strict majority, ``k // 2 + 1``).  The expectation under
        independence uses each checker's own error rate.
        """
        k = self.k
        thr = k // 2 + 1 if threshold is None else threshold
        if not 1 <= thr <= k:
            raise ValueError("threshold must be between 1 and k")
        obs = sum(1 for r in self._rows if sum(r) >= thr) / self.n
        all_obs = sum(1 for r in self._rows if sum(r) == k) / self.n
        ps = self.error_rates
        exp = _poisson_binomial_tail(ps, thr)
        all_exp = math.prod(ps)
        return MajorityStat(thr, obs, exp, all_obs, all_exp)

    # -- redundancy ----------------------------------------------------------

    def redundancy(self) -> dict:
        n = self.n
        out = {}
        for i in range(self.k):
            vals = []
            for j in range(self.k):
                if i == j:
                    continue
                p = _phi(self._pair_n11(i, j), self._cnt[i], self._cnt[j], n)
                if p is not None:
                    vals.append(p)
            out[self.names[i]] = sum(vals) / len(vals) if vals else None
        return out

    # -- resampling ----------------------------------------------------------

    def _subset(self, idx: Sequence[int]) -> "Panel":
        return Panel([self._rows[i] for i in idx], self.names)

    def _resample_indices(self, rng: random.Random) -> list:
        if self.clusters is None:
            return [rng.randrange(self.n) for _ in range(self.n)]
        groups: dict = {}
        for i, c in enumerate(self.clusters):
            groups.setdefault(c, []).append(i)
        keys = list(groups)
        idx: list = []
        for _ in range(len(keys)):
            idx.extend(groups[keys[rng.randrange(len(keys))]])
        return idx

    def bootstrap(
        self,
        stat: Callable[["Panel"], Optional[float]],
        n_boot: int = 1000,
        seed: int = 0,
        level: float = 0.95,
    ) -> Interval:
        """Percentile bootstrap of ``stat`` over items (or clusters, if given)."""
        rng = random.Random(seed)
        est = stat(self)
        draws = []
        for _ in range(n_boot):
            v = stat(self._subset(self._resample_indices(rng)))
            if v is not None:
                draws.append(v)
        if len(draws) < max(20, n_boot // 2):
            return Interval(est, None, None)
        draws.sort()
        a = (1 - level) / 2
        lo = draws[int(a * (len(draws) - 1))]
        hi = draws[int(math.ceil((1 - a) * (len(draws) - 1)))]
        return Interval(est, lo, hi)

    # -- hypothesis test -----------------------------------------------------

    def dependence_test(self, n_perm: int = 2000, seed: int = 0) -> Optional[float]:
        """One-sided permutation p-value for positive dependence between checkers.

        Statistic: mean pairwise phi.  Null: each checker's errors are shuffled
        across items independently, which keeps every checker's error rate fixed but
        breaks any link between checkers.  Items are treated as exchangeable, so
        with strongly clustered items read the result alongside the cluster
        bootstrap interval.
        """
        obs = self.mean_phi()
        if obs is None:
            return None
        rng = random.Random(seed)
        base = [[(r[j]) for r in self._rows] for j in range(self.k)]
        ge = 0
        for _ in range(n_perm):
            cols = [self._cols[0]]
            for j in range(1, self.k):
                col = base[j][:]
                rng.shuffle(col)
                cols.append(_pack(col))
            vals = []
            for i in range(self.k):
                for j in range(i + 1, self.k):
                    p = _phi((cols[i] & cols[j]).bit_count(), self._cnt[i], self._cnt[j], self.n)
                    if p is not None:
                        vals.append(p)
            stat = sum(vals) / len(vals)
            if stat >= obs - 1e-12:
                ge += 1
        return (ge + 1) / (n_perm + 1)

    # -- report --------------------------------------------------------------

    def warnings(self) -> list:
        w = []
        if self.n < 100:
            w.append(f"only {self.n} items: intervals and p-values are unstable below about 100.")
        for name, c in zip(self.names, self._cnt):
            if c == 0 or c == self.n:
                w.append(f"{name} is constant ({'never' if c == 0 else 'always'} wrong); it is ignored in correlations.")
            elif c < 5 or self.n - c < 5:
                w.append(f"{name} has very few {'errors' if c < 5 else 'correct answers'} ({min(c, self.n - c)}); its correlations are noisy.")
        if self.clusters is not None:
            w.append("clusters supplied: intervals resample clusters, but the permutation test treats items as exchangeable.")
        return w

    def report(self, n_perm: int = 2000, n_boot: int = 1000, seed: int = 0) -> Report:
        return Report(
            n_items=self.n,
            names=self.names,
            error_rates=self.error_rates,
            mean_phi=self.bootstrap(lambda p: p.mean_phi(), n_boot, seed),
            permutation_p=self.dependence_test(n_perm, seed),
            n_perm=n_perm,
            n_eff_participation=self.bootstrap(
                lambda p: p.effective_checkers("participation"), n_boot, seed
            ),
            n_eff_kish=self.bootstrap(lambda p: p.effective_checkers("kish"), n_boot, seed),
            majority=self.majority_vote(),
            pairs=self.pairwise(),
            redundancy=self.redundancy(),
            warnings=self.warnings(),
            cluster_bootstrap=self.clusters is not None,
        )
