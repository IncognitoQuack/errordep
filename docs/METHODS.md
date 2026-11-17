# Methods

Notation: n items, k checkers, `w_ij = 1` if checker j is wrong on item i. `n_a` is checker a's error count, `n_ab` the count of items both a and b get wrong.

**Phi.** `phi = (n * n_ab - n_a * n_b) / sqrt(n_a (n - n_a) n_b (n - n_b))`, the Pearson correlation of two binary columns. Undefined (reported as n/a, excluded from means) when a checker is constant.

**Expected both-wrong and lift.** Under independence `E[n_ab] = n_a * n_b / n`. Lift is `n_ab / E[n_ab]`.

**Conditional rates.** `P(b wrong | a wrong) = n_ab / n_a` and `P(b wrong | a right) = (n_b - n_ab) / (n - n_a)`. Their gap is the quantity behind "the check repeats the model's mistakes".

**Fisher test.** One-sided exact p for `X >= n_ab` with `X ~ Hypergeometric(n, n_a, n_b)`.

**Dependence permutation test.** Statistic: mean pairwise phi. Null: shuffle each checker's error column independently (the first is held fixed), preserving every error rate. `p = (1 + #{perm stat >= observed}) / (1 + n_perm)`.

**Effective checkers.**
- Kish: `m / (1 + (m - 1) * rho)` with `rho` the mean pairwise phi floored at 0 and m the number of non-constant checkers. This is the design effect for the average of m equally correlated checkers.
- Participation: `m^2 / sum_ij R_ij^2` for the correlation matrix R, equal to `(sum lambda)^2 / sum lambda^2`. Under equal correlation it is `m / (1 + (m - 1) rho^2)`, which is why it is always at least the Kish value. It counts independent dimensions of error rather than independent votes.

**Majority vote.** The panel is wrong when at least `t` checkers are wrong (default `k // 2 + 1`). The independence benchmark is the Poisson-binomial tail computed from each checker's own error rate. The all-wrong benchmark is the product of the error rates.

**Bootstrap.** Percentile intervals, resampling items (or whole clusters when a cluster id is given), 1000 resamples by default.
