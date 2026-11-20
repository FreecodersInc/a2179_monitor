# Nested CV, DE jobs: 462 rows, 207 positives (207 of them real), 5 folds x 12 repeats, seed 0, F1 ties: sweep, train recall >= 0.0

Hybrid mode (regex OR gate and score >= t). t, lam and min-max chosen on the training folds only. mean ± sd over repeats of the pooled test folds.

| channel | P | R | F1 | FP | FN | real P | real R | real FP | real FN |
|---|---|---|---|---|---|---|---|---|---|
| feat only | 0.824 ± 0.000 | 0.676 ± 0.000 | 0.743 ± 0.000 | 30.0 ± 0.0 | 67.0 ± 0.0 | 0.824 ± 0.000 | 0.676 ± 0.000 | 30.0 ± 0.0 | 67.0 ± 0.0 |
| tfidf+feat | 0.817 ± 0.003 | 0.669 ± 0.003 | 0.736 ± 0.003 | 31.0 ± 0.6 | 68.4 ± 0.8 | 0.817 ± 0.003 | 0.669 ± 0.003 | 31.0 ± 0.6 | 68.4 ± 0.8 |
| bm25 | 0.822 ± 0.004 | 0.663 ± 0.012 | 0.734 ± 0.007 | 29.7 ± 1.2 | 69.8 ± 2.5 | 0.822 ± 0.004 | 0.663 ± 0.012 | 29.7 ± 1.2 | 69.8 ± 2.5 |
| tfidf | 0.813 ± 0.001 | 0.668 ± 0.005 | 0.733 ± 0.003 | 31.8 ± 0.4 | 68.7 ± 1.0 | 0.813 ± 0.001 | 0.668 ± 0.005 | 31.8 ± 0.4 | 68.7 ± 1.0 |
| bm25+feat | 0.818 ± 0.007 | 0.662 ± 0.015 | 0.731 ± 0.008 | 30.6 ± 1.8 | 70.0 ± 3.1 | 0.818 ± 0.007 | 0.662 ± 0.015 | 30.6 ± 1.8 | 70.0 ± 3.1 |
| regex | 0.978 ± 0.000 | 0.435 ± 0.000 | 0.602 ± 0.000 | 2.0 ± 0.0 | 117.0 ± 0.0 | 0.978 ± 0.000 | 0.435 ± 0.000 | 2.0 ± 0.0 | 117.0 ± 0.0 |

Paired on identical folds (difference tfidf+feat minus other, negative FP = tfidf+feat better):

| pair | real FP diff mean ± sd | tfidf+feat better / tie / other better | F1 diff mean ± sd |
|---|---|---|---|
| tfidf+feat vs bm25+feat | 0.42 ± 2.02 | 5 / 2 / 5 | 0.0 ± 0.01 |
