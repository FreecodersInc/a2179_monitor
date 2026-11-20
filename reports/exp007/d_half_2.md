# Nested CV, DE jobs: 462 rows, 207 positives (207 of them real), 5 folds x 12 repeats, seed 0, F1 ties: sweep, train recall >= 0.0

Hybrid mode (regex OR gate and score >= t). t, lam and min-max chosen on the training folds only. mean ± sd over repeats of the pooled test folds.

| channel | P | R | F1 | FP | FN | real P | real R | real FP | real FN |
|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.863 ± 0.005 | 0.662 ± 0.000 | 0.750 ± 0.002 | 21.7 ± 0.9 | 70.0 ± 0.0 | 0.863 ± 0.005 | 0.662 ± 0.000 | 21.7 ± 0.9 | 70.0 ± 0.0 |
| bm25+feat | 0.863 ± 0.005 | 0.662 ± 0.000 | 0.750 ± 0.002 | 21.7 ± 0.9 | 70.0 ± 0.0 | 0.863 ± 0.005 | 0.662 ± 0.000 | 21.7 ± 0.9 | 70.0 ± 0.0 |
| feat only | 0.824 ± 0.000 | 0.676 ± 0.000 | 0.743 ± 0.000 | 30.0 ± 0.0 | 67.0 ± 0.0 | 0.824 ± 0.000 | 0.676 ± 0.000 | 30.0 ± 0.0 | 67.0 ± 0.0 |
| tfidf | 0.830 ± 0.005 | 0.661 ± 0.005 | 0.736 ± 0.004 | 27.9 ± 1.0 | 70.2 ± 1.1 | 0.830 ± 0.005 | 0.661 ± 0.005 | 27.9 ± 1.0 | 70.2 ± 1.1 |
| tfidf+feat | 0.830 ± 0.005 | 0.661 ± 0.005 | 0.736 ± 0.003 | 27.9 ± 1.0 | 70.2 ± 1.0 | 0.830 ± 0.005 | 0.661 ± 0.005 | 27.9 ± 1.0 | 70.2 ± 1.0 |
| regex | 0.978 ± 0.000 | 0.435 ± 0.000 | 0.602 ± 0.000 | 2.0 ± 0.0 | 117.0 ± 0.0 | 0.978 ± 0.000 | 0.435 ± 0.000 | 2.0 ± 0.0 | 117.0 ± 0.0 |

Paired on identical folds (difference tfidf+feat minus other, negative FP = tfidf+feat better):

| pair | real FP diff mean ± sd | tfidf+feat better / tie / other better | F1 diff mean ± sd |
|---|---|---|---|
| tfidf+feat vs bm25+feat | 6.25 ± 1.42 | 0 / 0 / 12 | -0.01 ± 0.0 |
