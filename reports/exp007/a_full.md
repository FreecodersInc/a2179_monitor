# Nested CV, A2179: 430 rows, 48 positives (17 of them real), 5 folds x 12 repeats, seed 0, F1 ties: sweep, train recall >= 1.0

Hybrid mode (regex OR gate and score >= t). t, lam and min-max chosen on the training folds only. mean ± sd over repeats of the pooled test folds.

| channel | P | R | F1 | FP | FN | real P | real R | real FP | real FN |
|---|---|---|---|---|---|---|---|---|---|
| regex | 0.939 ± 0.000 | 0.958 ± 0.000 | 0.948 ± 0.000 | 3.0 ± 0.0 | 2.0 ± 0.0 | 0.889 ± 0.000 | 0.941 ± 0.000 | 2.0 ± 0.0 | 1.0 ± 0.0 |
| tfidf+feat | 0.848 ± 0.013 | 0.977 ± 0.006 | 0.908 ± 0.007 | 8.4 ± 0.9 | 1.1 ± 0.3 | 0.696 ± 0.024 | 0.995 ± 0.016 | 7.4 ± 0.9 | 0.1 ± 0.3 |
| bm25+feat | 0.759 ± 0.018 | 0.977 ± 0.006 | 0.855 ± 0.011 | 14.9 ± 1.5 | 1.1 ± 0.3 | 0.568 ± 0.028 | 0.995 ± 0.016 | 12.9 ± 1.5 | 0.1 ± 0.3 |
| tfidf | 0.671 ± 0.012 | 0.977 ± 0.006 | 0.796 ± 0.007 | 23.0 ± 1.3 | 1.1 ± 0.3 | 0.435 ± 0.012 | 0.995 ± 0.016 | 22.0 ± 1.3 | 0.1 ± 0.3 |
| feat only | 0.623 ± 0.000 | 1.000 ± 0.000 | 0.768 ± 0.000 | 29.0 ± 0.0 | 0.0 ± 0.0 | 0.405 ± 0.000 | 1.000 ± 0.000 | 25.0 ± 0.0 | 0.0 ± 0.0 |
| bm25 | 0.578 ± 0.012 | 0.977 ± 0.006 | 0.726 ± 0.009 | 34.2 ± 1.8 | 1.1 ± 0.3 | 0.345 ± 0.011 | 0.995 ± 0.016 | 32.2 ± 1.8 | 0.1 ± 0.3 |

Paired on identical folds (difference tfidf+feat minus other, negative FP = tfidf+feat better):

| pair | real FP diff mean ± sd | tfidf+feat better / tie / other better | F1 diff mean ± sd |
|---|---|---|---|
| tfidf+feat vs bm25+feat | -5.5 ± 0.96 | 12 / 0 / 0 | 0.05 ± 0.01 |
