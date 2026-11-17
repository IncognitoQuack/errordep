import csv
import itertools
import json
import math
import os
import random
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO

from errordep import Panel
from errordep.cli import load_panel, main
from errordep.core import _fisher_greater, _phi, _poisson_binomial_tail


def independent(n, ps, seed):
    rng = random.Random(seed)
    return [[rng.random() < p for p in ps] for _ in range(n)]


def shared_difficulty(n, k, seed, base=0.1, hard=0.7, frac_hard=0.2):
    """Items are hard or easy for everyone, so errors are dependent."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        p = hard if rng.random() < frac_hard else base
        rows.append([rng.random() < p for _ in range(k)])
    return rows


class HandComputed(unittest.TestCase):
    def test_phi_and_counts(self):
        # 10 items; a wrong on 4, b wrong on 4, both wrong on 3.
        self.assertAlmostEqual(_phi(3, 4, 4, 10), (10 * 3 - 16) / math.sqrt(4 * 6 * 4 * 6))
        self.assertIsNone(_phi(0, 0, 4, 10))

    def test_fisher_matches_exhaustive_hypergeometric(self):
        n, na, nb = 12, 5, 6
        # exact tail by brute force over all placements
        items = range(n)
        total = ge = 0
        for A in itertools.combinations(items, na):
            for B in itertools.combinations(items, nb):
                total += 1
                ge += len(set(A) & set(B)) >= 4
        self.assertAlmostEqual(_fisher_greater(4, na, nb, n), ge / total, places=9)

    def test_pairwise_conditionals(self):
        wrong = [[1, 1]] * 3 + [[1, 0]] + [[0, 1]] + [[0, 0]] * 5
        p = Panel(wrong).pairwise()[0]
        self.assertEqual(p.both_wrong, 3)
        self.assertAlmostEqual(p.p_b_wrong_given_a_wrong, 3 / 4)
        self.assertAlmostEqual(p.p_b_wrong_given_a_right, 1 / 6)
        self.assertAlmostEqual(p.expected_both_wrong, 4 * 4 / 10)
        self.assertAlmostEqual(p.lift, 3 / 1.6)

    def test_poisson_binomial_exact(self):
        ps = [0.2, 0.5, 0.7]
        exact = sum(
            math.prod(p if b else 1 - p for p, b in zip(ps, bits))
            for bits in itertools.product([0, 1], repeat=3)
            if sum(bits) >= 2
        )
        self.assertAlmostEqual(_poisson_binomial_tail(ps, 2), exact)


class EffectiveCheckers(unittest.TestCase):
    def test_duplicate_checkers_count_as_one(self):
        base = independent(400, [0.3], 1)
        panel = Panel([[r[0], r[0]] for r in base])
        for method in ("participation", "kish"):
            self.assertAlmostEqual(panel.effective_checkers(method), 1.0, places=9)

    def test_independent_checkers_count_as_k(self):
        panel = Panel(independent(5000, [0.3] * 6, 2))
        self.assertGreater(panel.effective_checkers("participation"), 5.6)
        self.assertGreater(panel.effective_checkers("kish"), 5.0)

    def test_shared_difficulty_reduces_count(self):
        # Analytic values for the shared_difficulty model with k=6:
        # marginal p = 0.22, pair rate = 0.106, so rho = 0.3357;
        # Kish = 6/(1+5*rho) = 2.24, participation = 6/(1+5*rho^2) = 3.84.
        panel = Panel(shared_difficulty(20000, 6, 3))
        self.assertAlmostEqual(panel.effective_checkers("kish"), 2.24, delta=0.08)
        self.assertAlmostEqual(panel.effective_checkers("participation"), 3.84, delta=0.12)
        self.assertGreaterEqual(panel.effective_checkers("participation"), panel.effective_checkers("kish"))

    def test_constant_checker_is_ignored(self):
        rows = [[a, 0, b] for a, b in independent(300, [0.3, 0.3], 4)]
        panel = Panel(rows, names=["a", "const", "b"])
        self.assertAlmostEqual(panel.effective_checkers(), 2.0, delta=0.3)
        self.assertTrue(any("constant" in w for w in panel.warnings()))
        self.assertIsNone(panel.redundancy()["const"])


class DependenceTest(unittest.TestCase):
    def test_type_one_error_is_controlled(self):
        rejections = 0
        trials = 150
        for s in range(trials):
            panel = Panel(independent(120, [0.3, 0.4, 0.25], 1000 + s))
            if panel.dependence_test(n_perm=199, seed=s) <= 0.05:
                rejections += 1
        # 5% nominal; allow Monte Carlo slack.
        self.assertLessEqual(rejections / trials, 0.11)

    def test_power_against_shared_difficulty(self):
        panel = Panel(shared_difficulty(300, 4, 5))
        self.assertLess(panel.dependence_test(n_perm=499, seed=0), 0.01)

    def test_is_deterministic_given_seed(self):
        panel = Panel(shared_difficulty(150, 3, 6))
        self.assertEqual(panel.dependence_test(99, seed=7), panel.dependence_test(99, seed=7))


class MajorityVote(unittest.TestCase):
    def test_matches_independence_when_independent(self):
        panel = Panel(independent(40000, [0.3, 0.3, 0.3, 0.3, 0.3], 8))
        m = panel.majority_vote()
        self.assertAlmostEqual(m.observed, m.expected_if_independent, delta=0.01)
        self.assertAlmostEqual(m.ratio, 1.0, delta=0.12)

    def test_exceeds_independence_under_shared_difficulty(self):
        m = Panel(shared_difficulty(4000, 5, 9)).majority_vote()
        self.assertGreater(m.ratio, 1.5)
        self.assertGreater(m.all_wrong_observed, 3 * m.all_wrong_expected_if_independent)

    def test_threshold_bounds(self):
        with self.assertRaises(ValueError):
            Panel(independent(20, [0.5, 0.5], 0)).majority_vote(threshold=3)


class Bootstrap(unittest.TestCase):
    def test_interval_contains_estimate_and_is_reproducible(self):
        panel = Panel(shared_difficulty(300, 4, 10))
        a = panel.bootstrap(lambda p: p.mean_phi(), 300, seed=1)
        b = panel.bootstrap(lambda p: p.mean_phi(), 300, seed=1)
        self.assertEqual(a, b)
        self.assertLessEqual(a.lo, a.estimate)
        self.assertGreaterEqual(a.hi, a.estimate)

    def test_interval_narrows_with_more_items(self):
        small = Panel(shared_difficulty(150, 4, 11)).bootstrap(lambda p: p.mean_phi(), 300, 1)
        large = Panel(shared_difficulty(2400, 4, 11)).bootstrap(lambda p: p.mean_phi(), 300, 1)
        self.assertLess(large.hi - large.lo, small.hi - small.lo)

    def test_cluster_bootstrap_is_wider_when_items_repeat_within_clusters(self):
        base = shared_difficulty(120, 4, 12)
        rows = [r for r in base for _ in range(5)]
        clusters = [i for i in range(120) for _ in range(5)]
        item = Panel(rows).bootstrap(lambda p: p.mean_phi(), 400, 3)
        clus = Panel(rows, clusters=clusters).bootstrap(lambda p: p.mean_phi(), 400, 3)
        self.assertGreater(clus.hi - clus.lo, 1.5 * (item.hi - item.lo))


class Inputs(unittest.TestCase):
    def test_validation(self):
        with self.assertRaises(ValueError):
            Panel([[1]])
        with self.assertRaises(ValueError):
            Panel([[1, 0], [1]])
        with self.assertRaises(ValueError):
            Panel([[1, 0]], names=["x", "x"])
        with self.assertRaises(ValueError):
            Panel.from_verdicts([["a", "b"]], ["a", "b"])

    def test_from_verdicts_and_correct_agree(self):
        v = [["A", "B"], ["A", "A"], ["C", "C"]]
        gold = ["A", "A", "B"]
        p1 = Panel.from_verdicts(v, gold)
        p2 = Panel.from_correct([[x == g for x in row] for row, g in zip(v, gold)])
        self.assertEqual(p1._rows, p2._rows)

    def test_small_sample_warning(self):
        self.assertTrue(any("only" in w for w in Panel(independent(30, [0.3, 0.3], 0)).warnings()))


class CLI(unittest.TestCase):
    def _write(self, rows, header):
        fh = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="")
        csv.writer(fh).writerows([header] + rows)
        fh.close()
        self.addCleanup(os.unlink, fh.name)
        return fh.name

    def test_verdicts_against_gold(self):
        rng = random.Random(0)
        rows = []
        for i in range(200):
            gold = rng.choice("ABCD")
            hard = rng.random() < 0.3
            ans = [("X" if hard and rng.random() < 0.8 else gold) for _ in range(3)]
            rows.append([i, gold] + ans)
        path = self._write(rows, ["id", "gold", "m1", "m2", "m3"])
        out = StringIO()
        with redirect_stdout(out):
            rc = main([path, "--gold", "gold", "--ignore", "id", "--json", "--permutations", "199", "--bootstrap", "100"])
        self.assertEqual(rc, 0)
        data = json.loads(out.getvalue())
        self.assertEqual(data["n_checkers"], 3)
        self.assertLess(data["dependence_permutation_p"], 0.01)
        self.assertGreater(data["majority_wrong"]["ratio"], 1.2)

    def test_text_report_renders(self):
        rows = [[i, int(i % 3 == 0), int(i % 3 == 0 or i % 7 == 0), int(i % 5 == 0)] for i in range(120)]
        path = self._write(rows, ["id", "a", "b", "c"])
        out = StringIO()
        with redirect_stdout(out):
            main([path, "--as-wrong", "--ignore", "id", "--permutations", "99", "--bootstrap", "50"])
        self.assertIn("Effective number of independent checkers", out.getvalue())

    def test_bad_flag_is_a_clean_error(self):
        path = self._write([[1, "maybe"], [0, "1"]], ["a", "b"])
        with self.assertRaises(SystemExit):
            load_panel(path, as_wrong=True)

    def test_requires_a_mode(self):
        path = self._write([[1, 0]], ["a", "b"])
        with self.assertRaises(SystemExit):
            load_panel(path)


if __name__ == "__main__":
    unittest.main()
