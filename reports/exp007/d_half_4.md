# Nested CV, DE jobs: 462 rows, 207 positives (207 of them real), 5 folds x 12 repeats, seed 0, F1 ties: sweep, train recall >= 0.0

Hybrid mode (regex OR gate and score >= t). t, lam and min-max chosen on the training folds only. mean ± sd over repeats of the pooled test folds.

| channel | P | R | F1 | FP | FN | real P | real R | real FP | real FN |
|---|---|---|---|---|---|---|---|---|---|
| feat only | 0.824 ± 0.000 | 0.676 ± 0.000 | 0.743 ± 0.000 | 30.0 ± 0.0 | 67.0 ± 0.0 | 0.824 ± 0.000 | 0.676 ± 0.000 | 30.0 ± 0.0 | 67.0 ± 0.0 |
| tfidf+feat | 0.816 ± 0.003 | 0.661 ± 0.006 | 0.731 ± 0.004 | 30.8 ± 0.7 | 70.1 ± 1.3 | 0.816 ± 0.003 | 0.661 ± 0.006 | 30.8 ± 0.7 | 70.1 ± 1.3 |
| tfidf | 0.812 ± 0.002 | 0.661 ± 0.006 | 0.729 ± 0.004 | 31.7 ± 0.5 | 70.1 ± 1.3 | 0.812 ± 0.002 | 0.661 ± 0.006 | 31.7 ± 0.5 | 70.1 ± 1.3 |
| bm25 | 0.830 ± 0.015 | 0.647 ± 0.018 | 0.726 ± 0.007 | 27.6 ± 3.6 | 73.2 ± 3.7 | 0.830 ± 0.015 | 0.647 ± 0.018 | 27.6 ± 3.6 | 73.2 ± 3.7 |
| bm25+feat | 0.838 ± 0.015 | 0.640 ± 0.020 | 0.725 ± 0.009 | 25.8 ± 3.6 | 74.5 ± 4.1 | 0.838 ± 0.015 | 0.640 ± 0.020 | 25.8 ± 3.6 | 74.5 ± 4.1 |
| regex | 0.978 ± 0.000 | 0.435 ± 0.000 | 0.602 ± 0.000 | 2.0 ± 0.0 | 117.0 ± 0.0 | 0.978 ± 0.000 | 0.435 ± 0.000 | 2.0 ± 0.0 | 117.0 ± 0.0 |

Paired on identical folds (difference tfidf+feat minus other, negative FP = tfidf+feat better):

| pair | real FP diff mean ± sd | tfidf+feat better / tie / other better | F1 diff mean ± sd |
|---|---|---|---|
| tfidf+feat vs bm25+feat | 5 ± 3.76 | 0 / 3 / 9 | 0.01 ± 0.01 |
