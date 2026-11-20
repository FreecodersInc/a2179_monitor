# Nested CV, DE jobs: 462 rows, 207 positives (207 of them real), 5 folds x 12 repeats, seed 0, F1 ties: sweep, train recall >= 0.0

Hybrid mode (regex OR gate and score >= t). t, lam and min-max chosen on the training folds only. mean ± sd over repeats of the pooled test folds.

| channel | P | R | F1 | FP | FN | real P | real R | real FP | real FN |
|---|---|---|---|---|---|---|---|---|---|
| feat only | 0.824 ± 0.000 | 0.676 ± 0.000 | 0.743 ± 0.000 | 30.0 ± 0.0 | 67.0 ± 0.0 | 0.824 ± 0.000 | 0.676 ± 0.000 | 30.0 ± 0.0 | 67.0 ± 0.0 |
| bm25+feat | 0.818 ± 0.005 | 0.668 ± 0.002 | 0.735 ± 0.002 | 30.8 ± 1.1 | 68.7 ± 0.5 | 0.818 ± 0.005 | 0.668 ± 0.002 | 30.8 ± 1.1 | 68.7 ± 0.5 |
| tfidf+feat | 0.817 ± 0.003 | 0.668 ± 0.006 | 0.734 ± 0.004 | 31.1 ± 0.5 | 68.8 ± 1.2 | 0.817 ± 0.003 | 0.668 ± 0.006 | 31.1 ± 0.5 | 68.8 ± 1.2 |
| bm25 | 0.817 ± 0.005 | 0.667 ± 0.001 | 0.734 ± 0.002 | 31.0 ± 1.0 | 68.9 ± 0.3 | 0.817 ± 0.005 | 0.667 ± 0.001 | 31.0 ± 1.0 | 68.9 ± 0.3 |
| dense:minilm+feat | 0.820 ± 0.005 | 0.664 ± 0.003 | 0.734 ± 0.002 | 30.1 ± 1.0 | 69.6 ± 0.6 | 0.820 ± 0.005 | 0.664 ± 0.003 | 30.1 ± 1.0 | 69.6 ± 0.6 |
| tfidf | 0.813 ± 0.003 | 0.665 ± 0.005 | 0.731 ± 0.004 | 31.8 ± 0.6 | 69.4 ± 1.0 | 0.813 ± 0.003 | 0.665 ± 0.005 | 31.8 ± 0.6 | 69.4 ± 1.0 |
| dense:minilm | 0.811 ± 0.003 | 0.663 ± 0.010 | 0.730 ± 0.007 | 32.0 ± 0.0 | 69.6 ± 2.1 | 0.811 ± 0.003 | 0.663 ± 0.010 | 32.0 ± 0.0 | 69.6 ± 2.1 |
| regex | 0.978 ± 0.000 | 0.435 ± 0.000 | 0.602 ± 0.000 | 2.0 ± 0.0 | 117.0 ± 0.0 | 0.978 ± 0.000 | 0.435 ± 0.000 | 2.0 ± 0.0 | 117.0 ± 0.0 |

Paired on identical folds (difference tfidf+feat minus other, negative FP = tfidf+feat better):

| pair | real FP diff mean ± sd | tfidf+feat better / tie / other better | F1 diff mean ± sd |
|---|---|---|---|
| tfidf+feat vs bm25+feat | 0.25 ± 0.83 | 2 / 6 / 4 | -0.0 ± 0.0 |
| tfidf+feat vs dense:minilm+feat | 1 ± 0.82 | 0 / 3 / 9 | 0.0 ± 0.0 |
