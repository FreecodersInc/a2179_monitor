# Nested CV, A2179: 430 rows, 48 positives (17 of them real), 5 folds x 12 repeats, seed 0, F1 ties: sweep, train recall >= 1.0

Hybrid mode (regex OR gate and score >= t). t, lam and min-max chosen on the training folds only. mean ± sd over repeats of the pooled test folds.

| channel | P | R | F1 | FP | FN | real P | real R | real FP | real FN |
|---|---|---|---|---|---|---|---|---|---|
| regex | 0.939 ± 0.000 | 0.958 ± 0.000 | 0.948 ± 0.000 | 3.0 ± 0.0 | 2.0 ± 0.0 | 0.889 ± 0.000 | 0.941 ± 0.000 | 2.0 ± 0.0 | 1.0 ± 0.0 |
| tfidf+feat | 0.773 ± 0.010 | 0.977 ± 0.006 | 0.863 ± 0.005 | 13.8 ± 0.8 | 1.1 ± 0.3 | 0.610 ± 0.016 | 0.995 ± 0.016 | 10.8 ± 0.8 | 0.1 ± 0.3 |
| bm25+feat | 0.740 ± 0.016 | 0.977 ± 0.006 | 0.842 ± 0.008 | 16.5 ± 1.4 | 1.1 ± 0.3 | 0.526 ± 0.027 | 0.941 ± 0.000 | 14.5 ± 1.4 | 1.0 ± 0.0 |
| feat only | 0.623 ± 0.000 | 1.000 ± 0.000 | 0.768 ± 0.000 | 29.0 ± 0.0 | 0.0 ± 0.0 | 0.405 ± 0.000 | 1.000 ± 0.000 | 25.0 ± 0.0 | 0.0 ± 0.0 |
| tfidf | 0.599 ± 0.009 | 0.977 ± 0.006 | 0.743 ± 0.008 | 31.4 ± 1.2 | 1.1 ± 0.3 | 0.411 ± 0.010 | 0.995 ± 0.016 | 24.2 ± 0.9 | 0.1 ± 0.3 |
| bm25 | 0.561 ± 0.011 | 0.977 ± 0.006 | 0.713 ± 0.008 | 36.8 ± 1.8 | 1.1 ± 0.3 | 0.316 ± 0.013 | 0.941 ± 0.000 | 34.8 ± 1.8 | 1.0 ± 0.0 |

Paired on identical folds (difference tfidf+feat minus other, negative FP = tfidf+feat better):

| pair | real FP diff mean ± sd | tfidf+feat better / tie / other better | F1 diff mean ± sd |
|---|---|---|---|
| tfidf+feat vs bm25+feat | -3.67 ± 1.03 | 12 / 0 / 0 | 0.02 ± 0.01 |
