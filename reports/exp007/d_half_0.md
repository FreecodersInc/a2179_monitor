# Nested CV, DE jobs: 462 rows, 207 positives (207 of them real), 5 folds x 12 repeats, seed 0, F1 ties: sweep, train recall >= 0.0

Hybrid mode (regex OR gate and score >= t). t, lam and min-max chosen on the training folds only. mean ± sd over repeats of the pooled test folds.

| channel | P | R | F1 | FP | FN | real P | real R | real FP | real FN |
|---|---|---|---|---|---|---|---|---|---|
| feat only | 0.824 ± 0.000 | 0.676 ± 0.000 | 0.743 ± 0.000 | 30.0 ± 0.0 | 67.0 ± 0.0 | 0.824 ± 0.000 | 0.676 ± 0.000 | 30.0 ± 0.0 | 67.0 ± 0.0 |
| tfidf+feat | 0.817 ± 0.002 | 0.668 ± 0.003 | 0.735 ± 0.002 | 31.2 ± 0.4 | 68.7 ± 0.6 | 0.817 ± 0.002 | 0.668 ± 0.003 | 31.2 ± 0.4 | 68.7 ± 0.6 |
| bm25+feat | 0.819 ± 0.005 | 0.665 ± 0.003 | 0.734 ± 0.002 | 30.4 ± 1.2 | 69.4 ± 0.6 | 0.819 ± 0.005 | 0.665 ± 0.003 | 30.4 ± 1.2 | 69.4 ± 0.6 |
| tfidf | 0.812 ± 0.001 | 0.668 ± 0.003 | 0.733 ± 0.002 | 32.0 ± 0.0 | 68.7 ± 0.6 | 0.812 ± 0.001 | 0.668 ± 0.003 | 32.0 ± 0.0 | 68.7 ± 0.6 |
| bm25 | 0.815 ± 0.004 | 0.665 ± 0.003 | 0.732 ± 0.002 | 31.3 ± 1.0 | 69.3 ± 0.6 | 0.815 ± 0.004 | 0.665 ± 0.003 | 31.3 ± 1.0 | 69.3 ± 0.6 |
| regex | 0.978 ± 0.000 | 0.435 ± 0.000 | 0.602 ± 0.000 | 2.0 ± 0.0 | 117.0 ± 0.0 | 0.978 ± 0.000 | 0.435 ± 0.000 | 2.0 ± 0.0 | 117.0 ± 0.0 |

Paired on identical folds (difference tfidf+feat minus other, negative FP = tfidf+feat better):

| pair | real FP diff mean ± sd | tfidf+feat better / tie / other better | F1 diff mean ± sd |
|---|---|---|---|
| tfidf+feat vs bm25+feat | 0.75 ± 1.01 | 0 / 7 / 5 | 0.0 ± 0.0 |
