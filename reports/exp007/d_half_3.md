# Nested CV, DE jobs: 462 rows, 207 positives (207 of them real), 5 folds x 12 repeats, seed 0, F1 ties: sweep, train recall >= 0.0

Hybrid mode (regex OR gate and score >= t). t, lam and min-max chosen on the training folds only. mean ± sd over repeats of the pooled test folds.

| channel | P | R | F1 | FP | FN | real P | real R | real FP | real FN |
|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.868 ± 0.000 | 0.667 ± 0.001 | 0.754 ± 0.001 | 21.0 ± 0.0 | 69.1 ± 0.3 | 0.868 ± 0.000 | 0.667 ± 0.001 | 21.0 ± 0.0 | 69.1 ± 0.3 |
| bm25+feat | 0.864 ± 0.002 | 0.667 ± 0.001 | 0.752 ± 0.001 | 21.8 ± 0.4 | 69.1 ± 0.3 | 0.864 ± 0.002 | 0.667 ± 0.001 | 21.8 ± 0.4 | 69.1 ± 0.3 |
| feat only | 0.824 ± 0.000 | 0.676 ± 0.000 | 0.743 ± 0.000 | 30.0 ± 0.0 | 67.0 ± 0.0 | 0.824 ± 0.000 | 0.676 ± 0.000 | 30.0 ± 0.0 | 67.0 ± 0.0 |
| tfidf | 0.840 ± 0.009 | 0.646 ± 0.004 | 0.730 ± 0.003 | 25.5 ± 1.8 | 73.2 ± 0.9 | 0.840 ± 0.009 | 0.646 ± 0.004 | 25.5 ± 1.8 | 73.2 ± 0.9 |
| tfidf+feat | 0.831 ± 0.006 | 0.648 ± 0.005 | 0.728 ± 0.003 | 27.2 ± 1.3 | 72.8 ± 1.1 | 0.831 ± 0.006 | 0.648 ± 0.005 | 27.2 ± 1.3 | 72.8 ± 1.1 |
| regex | 0.978 ± 0.000 | 0.435 ± 0.000 | 0.602 ± 0.000 | 2.0 ± 0.0 | 117.0 ± 0.0 | 0.978 ± 0.000 | 0.435 ± 0.000 | 2.0 ± 0.0 | 117.0 ± 0.0 |

Paired on identical folds (difference tfidf+feat minus other, negative FP = tfidf+feat better):

| pair | real FP diff mean ± sd | tfidf+feat better / tie / other better | F1 diff mean ± sd |
|---|---|---|---|
| tfidf+feat vs bm25+feat | 5.5 ± 1.55 | 0 / 0 / 12 | -0.02 ± 0.0 |
