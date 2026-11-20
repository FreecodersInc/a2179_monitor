# Nested CV, A2179: 430 rows, 48 positives (17 of them real), 5 folds x 12 repeats, seed 0, F1 ties: sweep, train recall >= 1.0

Hybrid mode (regex OR gate and score >= t). t, lam and min-max chosen on the training folds only. mean ± sd over repeats of the pooled test folds.

| channel | P | R | F1 | FP | FN | real P | real R | real FP | real FN |
|---|---|---|---|---|---|---|---|---|---|
| regex | 0.939 ± 0.000 | 0.958 ± 0.000 | 0.948 ± 0.000 | 3.0 ± 0.0 | 2.0 ± 0.0 | 0.889 ± 0.000 | 0.941 ± 0.000 | 2.0 ± 0.0 | 1.0 ± 0.0 |
| bm25+feat | 0.873 ± 0.020 | 0.977 ± 0.006 | 0.923 ± 0.013 | 6.8 ± 1.2 | 1.1 ± 0.3 | 0.746 ± 0.041 | 0.995 ± 0.016 | 5.8 ± 1.2 | 0.1 ± 0.3 |
| bm25 | 0.853 ± 0.012 | 0.977 ± 0.006 | 0.911 ± 0.008 | 8.1 ± 0.8 | 1.1 ± 0.3 | 0.705 ± 0.024 | 0.995 ± 0.016 | 7.1 ± 0.8 | 0.1 ± 0.3 |
| tfidf+feat | 0.803 ± 0.027 | 0.977 ± 0.006 | 0.881 ± 0.016 | 11.6 ± 1.9 | 1.1 ± 0.3 | 0.618 ± 0.047 | 0.995 ± 0.016 | 10.6 ± 1.9 | 0.1 ± 0.3 |
| feat only | 0.623 ± 0.000 | 1.000 ± 0.000 | 0.768 ± 0.000 | 29.0 ± 0.0 | 0.0 ± 0.0 | 0.405 ± 0.000 | 1.000 ± 0.000 | 25.0 ± 0.0 | 0.0 ± 0.0 |
| tfidf | 0.620 ± 0.023 | 0.977 ± 0.006 | 0.758 ± 0.017 | 28.9 ± 2.9 | 1.1 ± 0.3 | 0.379 ± 0.023 | 0.995 ± 0.016 | 27.9 ± 2.9 | 0.1 ± 0.3 |

Paired on identical folds (difference tfidf+feat minus other, negative FP = tfidf+feat better):

| pair | real FP diff mean ± sd | tfidf+feat better / tie / other better | F1 diff mean ± sd |
|---|---|---|---|
| tfidf+feat vs bm25+feat | 4.75 ± 2.09 | 0 / 0 / 12 | -0.04 ± 0.02 |
