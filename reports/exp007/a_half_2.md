# Nested CV, A2179: 430 rows, 48 positives (17 of them real), 5 folds x 12 repeats, seed 0, F1 ties: sweep, train recall >= 1.0

Hybrid mode (regex OR gate and score >= t). t, lam and min-max chosen on the training folds only. mean ± sd over repeats of the pooled test folds.

| channel | P | R | F1 | FP | FN | real P | real R | real FP | real FN |
|---|---|---|---|---|---|---|---|---|---|
| regex | 0.939 ± 0.000 | 0.958 ± 0.000 | 0.948 ± 0.000 | 3.0 ± 0.0 | 2.0 ± 0.0 | 0.889 ± 0.000 | 0.941 ± 0.000 | 2.0 ± 0.0 | 1.0 ± 0.0 |
| bm25+feat | 0.815 ± 0.018 | 0.977 ± 0.006 | 0.889 ± 0.011 | 10.7 ± 1.2 | 1.1 ± 0.3 | 0.663 ± 0.033 | 0.995 ± 0.016 | 8.7 ± 1.2 | 0.1 ± 0.3 |
| bm25 | 0.796 ± 0.015 | 0.977 ± 0.006 | 0.878 ± 0.009 | 12.0 ± 1.2 | 1.1 ± 0.3 | 0.630 ± 0.026 | 0.995 ± 0.016 | 10.0 ± 1.2 | 0.1 ± 0.3 |
| tfidf+feat | 0.796 ± 0.024 | 0.977 ± 0.006 | 0.877 ± 0.014 | 12.1 ± 1.8 | 1.1 ± 0.3 | 0.627 ± 0.039 | 0.995 ± 0.016 | 10.2 ± 1.7 | 0.1 ± 0.3 |
| feat only | 0.623 ± 0.000 | 1.000 ± 0.000 | 0.768 ± 0.000 | 29.0 ± 0.0 | 0.0 ± 0.0 | 0.405 ± 0.000 | 1.000 ± 0.000 | 25.0 ± 0.0 | 0.0 ± 0.0 |
| tfidf | 0.573 ± 0.020 | 0.977 ± 0.006 | 0.723 ± 0.015 | 35.0 ± 2.9 | 1.1 ± 0.3 | 0.370 ± 0.021 | 0.995 ± 0.016 | 29.0 ± 2.8 | 0.1 ± 0.3 |

Paired on identical folds (difference tfidf+feat minus other, negative FP = tfidf+feat better):

| pair | real FP diff mean ± sd | tfidf+feat better / tie / other better | F1 diff mean ± sd |
|---|---|---|---|
| tfidf+feat vs bm25+feat | 1.5 ± 1.32 | 1 / 2 / 9 | -0.01 ± 0.01 |
