# Nested CV, A2179: 430 rows, 48 positives (17 of them real), 5 folds x 12 repeats, seed 0, F1 ties: sweep, train recall >= 1.0

Hybrid mode (regex OR gate and score >= t). t, lam and min-max chosen on the training folds only. mean ± sd over repeats of the pooled test folds.

| channel | P | R | F1 | FP | FN | real P | real R | real FP | real FN |
|---|---|---|---|---|---|---|---|---|---|
| regex | 0.939 ± 0.000 | 0.958 ± 0.000 | 0.948 ± 0.000 | 3.0 ± 0.0 | 2.0 ± 0.0 | 0.889 ± 0.000 | 0.941 ± 0.000 | 2.0 ± 0.0 | 1.0 ± 0.0 |
| tfidf+feat | 0.815 ± 0.025 | 0.977 ± 0.006 | 0.889 ± 0.015 | 10.8 ± 1.7 | 1.1 ± 0.3 | 0.637 ± 0.044 | 0.995 ± 0.016 | 9.8 ± 1.7 | 0.1 ± 0.3 |
| tfidf | 0.763 ± 0.020 | 0.977 ± 0.006 | 0.857 ± 0.013 | 14.6 ± 1.6 | 1.1 ± 0.3 | 0.556 ± 0.031 | 0.995 ± 0.016 | 13.6 ± 1.6 | 0.1 ± 0.3 |
| bm25+feat | 0.714 ± 0.020 | 0.977 ± 0.006 | 0.825 ± 0.013 | 18.8 ± 1.9 | 1.1 ± 0.3 | 0.517 ± 0.030 | 0.995 ± 0.016 | 15.9 ± 1.9 | 0.1 ± 0.3 |
| feat only | 0.623 ± 0.000 | 1.000 ± 0.000 | 0.768 ± 0.000 | 29.0 ± 0.0 | 0.0 ± 0.0 | 0.405 ± 0.000 | 1.000 ± 0.000 | 25.0 ± 0.0 | 0.0 ± 0.0 |
| bm25 | 0.297 ± 0.006 | 0.977 ± 0.006 | 0.456 ± 0.007 | 111.2 ± 3.2 | 1.1 ± 0.3 | 0.142 ± 0.003 | 0.995 ± 0.016 | 102.4 ± 2.6 | 0.1 ± 0.3 |

Paired on identical folds (difference tfidf+feat minus other, negative FP = tfidf+feat better):

| pair | real FP diff mean ± sd | tfidf+feat better / tie / other better | F1 diff mean ± sd |
|---|---|---|---|
| tfidf+feat vs bm25+feat | -6.17 ± 0.99 | 12 / 0 / 0 | 0.06 ± 0.01 |
