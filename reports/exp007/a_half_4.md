# Nested CV, A2179: 430 rows, 48 positives (17 of them real), 5 folds x 12 repeats, seed 0, F1 ties: sweep, train recall >= 1.0

Hybrid mode (regex OR gate and score >= t). t, lam and min-max chosen on the training folds only. mean ± sd over repeats of the pooled test folds.

| channel | P | R | F1 | FP | FN | real P | real R | real FP | real FN |
|---|---|---|---|---|---|---|---|---|---|
| regex | 0.939 ± 0.000 | 0.958 ± 0.000 | 0.948 ± 0.000 | 3.0 ± 0.0 | 2.0 ± 0.0 | 0.889 ± 0.000 | 0.941 ± 0.000 | 2.0 ± 0.0 | 1.0 ± 0.0 |
| bm25+feat | 0.822 ± 0.011 | 0.977 ± 0.006 | 0.893 ± 0.006 | 10.2 ± 0.8 | 1.1 ± 0.3 | 0.673 ± 0.021 | 0.995 ± 0.016 | 8.2 ± 0.8 | 0.1 ± 0.3 |
| tfidf+feat | 0.811 ± 0.025 | 0.977 ± 0.006 | 0.886 ± 0.015 | 11.0 ± 1.8 | 1.1 ± 0.3 | 0.631 ± 0.044 | 0.995 ± 0.016 | 10.0 ± 1.8 | 0.1 ± 0.3 |
| bm25 | 0.799 ± 0.013 | 0.977 ± 0.006 | 0.879 ± 0.007 | 11.8 ± 1.0 | 1.1 ± 0.3 | 0.632 ± 0.023 | 0.995 ± 0.016 | 9.9 ± 1.0 | 0.1 ± 0.3 |
| tfidf | 0.763 ± 0.018 | 0.977 ± 0.006 | 0.857 ± 0.011 | 14.6 ± 1.4 | 1.1 ± 0.3 | 0.556 ± 0.027 | 0.995 ± 0.016 | 13.6 ± 1.4 | 0.1 ± 0.3 |
| feat only | 0.623 ± 0.000 | 1.000 ± 0.000 | 0.768 ± 0.000 | 29.0 ± 0.0 | 0.0 ± 0.0 | 0.405 ± 0.000 | 1.000 ± 0.000 | 25.0 ± 0.0 | 0.0 ± 0.0 |

Paired on identical folds (difference tfidf+feat minus other, negative FP = tfidf+feat better):

| pair | real FP diff mean ± sd | tfidf+feat better / tie / other better | F1 diff mean ± sd |
|---|---|---|---|
| tfidf+feat vs bm25+feat | 1.75 ± 1.23 | 1 / 1 / 10 | -0.01 ± 0.01 |
