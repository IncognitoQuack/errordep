"""Generate examples/simulated_panel.csv: a simulated panel of four LLM judges.

Items are hard or easy for everyone (shared difficulty), and the first two judges
also share an extra source of mistakes (think: same base model).  The third and
fourth judges err independently of the rest, apart from that shared difficulty.
All data is synthetic.

Run:  python3 examples/make_example.py
"""
import csv
import os
import random

N = 800
rng = random.Random(2026)
rows = []
for i in range(N):
    hard = rng.random() < 0.25
    base = 0.55 if hard else 0.06
    shared_pair = rng.random() < (0.30 if hard else 0.05)  # affects judges 1 and 2 together
    wrong = [
        shared_pair or rng.random() < base * 0.6,
        shared_pair or rng.random() < base * 0.6,
        rng.random() < base,
        rng.random() < base * 1.2,
    ]
    rows.append([f"item-{i:04d}"] + [int(not w) for w in wrong])  # 1 = judge was correct

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "simulated_panel.csv")
with open(path, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["id", "judge_a", "judge_b", "judge_c", "judge_d"])
    w.writerows(rows)
print(f"wrote {path}")
