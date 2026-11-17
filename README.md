# errordep

**If your LLM judges, verifiers or models all agree, is that several pieces of evidence or one piece counted several times?**

`errordep` answers that from a table of *who was wrong on which item*. It has no dependencies (Python 3.10+, standard library only) and ships a command-line tool and a small API.

Majority voting, "ask a second model to check", self-consistency and judge panels all assume the checkers make roughly independent mistakes. Often they do not: strong models share training data, and hard items are hard for everyone. When errors are dependent, a panel of 5 can be worth far fewer than 5 votes, and the confidence you read off the agreement is too high.

## Example (simulated data)

A simulated panel of four judges on 800 items (`examples/make_example.py` generates it; all data is synthetic). Items are hard or easy for everyone, and judges A and B also share an extra source of mistakes, as two models from the same family might:

```sh
errordep examples/simulated_panel.csv --as-correct --ignore id
```

```
Dependence between checkers' errors: mean pairwise phi = 0.360 (95% CI 0.310 to 0.409)
  permutation test for dependence (one-sided, 5000 permutations): p = 0.0002

Effective number of independent checkers (out of 4)
  Kish (independent votes)   1.92 (95% CI 1.80 to 2.07)

Panel wrong when >= 3 of 4 checkers are wrong
  observed 10.9% vs 2.4% expected if errors were independent (4.4x)

  judge_a + judge_b: both wrong 106 (expected 30.4), lift 3.49
```

Four judges were worth 1.92 independent votes, and a majority-vote panel was wrong 4.4 times as often as independence would predict. The pair built to share errors (A and B) shows the highest lift. (Intervals: 5000 permutations, 2000 bootstrap resamples, seed 0.)

## Install and run

```sh
pip install .                      # or: PYTHONPATH=src python3 -m errordep ...
errordep panel.csv --gold truth --ignore id          # columns hold each checker's answer
errordep flags.csv --as-correct                       # columns are 1/0 "was correct"
errordep flags.csv --as-wrong --cluster-col doc_id    # resample whole documents
errordep panel.csv --gold truth --json
```

Input is one row per item and one column per checker. Use `--gold COLUMN` (answers are compared as text with the gold column), `--as-correct`, or `--as-wrong`. `--cluster-col` makes the bootstrap resample groups of items that share a source.

## What it reports

| Output | Meaning |
|---|---|
| Pairwise table | both-wrong count vs expected under independence, lift, phi, `P(b wrong given a wrong)` vs `P(b wrong given a right)`, one-sided Fisher p |
| Dependence test | permutation p-value for the mean pairwise phi (each checker's errors shuffled independently, so error rates are kept) |
| Effective checkers | Kish design-effect count `m / (1 + (m-1) * mean phi)` (headline), and the participation ratio of the error-correlation matrix, with bootstrap intervals |
| Majority vote | how often the panel is wrong vs what independence predicts from each checker's own error rate (exact Poisson-binomial), plus the all-wrong rate |
| Redundancy | each checker's mean phi with the others |

```python
from errordep import Panel

panel = Panel.from_verdicts(answers, gold, names=["gpt", "claude", "llama"])
rep = panel.report()
print(rep)                      # text report
rep.to_dict()                   # JSON-ready
panel.effective_checkers()      # Kish count
panel.majority_vote().ratio     # observed / expected-if-independent
```

## What it does not tell you

- It needs **ground truth**. It measures dependence in errors, not in opinions.
- Dependence includes **shared item difficulty**. A small effective count means the checkers do not add independent evidence; it does not say whether the cause is shared training data, shared prompts or simply hard items.
- With few errors the correlations are noisy. The tool warns below 100 items or fewer than 5 errors per checker.
- The permutation test treats items as exchangeable. With strongly clustered items rely on the cluster-bootstrap intervals.
- It is a diagnostic, not a correction. It tells you that the votes are not independent; it does not turn them into calibrated probabilities.

See [docs/METHODS.md](docs/METHODS.md) for the formulas.

## Related work

The idea that LLM judges share errors is established, and this package makes the measurement easy to run rather than claiming to discover it:

- Goel et al., *Great Models Think Alike and this Undermines AI Oversight* (CAPA similarity metric); an implementation is at [model-similarity/lm-similarity](https://github.com/model-similarity/lm-similarity).
- *Nine Judges, Two Effective Votes: Correlated Errors Undermine LLM Evaluation Panels* ([arXiv:2605.29800](https://arxiv.org/abs/2605.29800)): effective panel size from correlated errors.
- *Agreement Overstates Evidence: Error Dependence in LLM Judge Consensus* ([arXiv:2609.22512](https://arxiv.org/abs/2609.22512)).
- Miller, *Adding Error Bars to Evals* ([arXiv:2411.00640](https://arxiv.org/abs/2411.00640)) for the error-bar side of eval statistics; pair it with this package.

## Development

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -t . -v
```

The tests include hand-computed cases, an exhaustive check of the Fisher test, analytic values for a shared-difficulty simulation, and a type-I-error check of the permutation test (150 simulated null panels).

## License

MIT. See [LICENSE](LICENSE).
