# Nested CV, DE jobs: 462 rows, 213 positives (213 of them real), 5 folds x 12 repeats, seed 0, F1 ties: sweep, train recall >= 0.0

Hybrid mode (regex OR gate and score >= t). t, lam and min-max chosen on the training folds only. mean ± sd over repeats of the pooled test folds.

| channel | P | R | F1 | FP | FN | real P | real R | real FP | real FN |
|---|---|---|---|---|---|---|---|---|---|
| feat only | 0.829 ± 0.000 | 0.934 ± 0.000 | 0.879 ± 0.000 | 41.0 ± 0.0 | 14.0 ± 0.0 | 0.829 ± 0.000 | 0.934 ± 0.000 | 41.0 ± 0.0 | 14.0 ± 0.0 |
| bm25+feat | 0.838 ± 0.005 | 0.920 ± 0.003 | 0.877 ± 0.003 | 37.9 ± 1.4 | 16.9 ± 0.6 | 0.838 ± 0.005 | 0.920 ± 0.003 | 37.9 ± 1.4 | 16.9 ± 0.6 |
| tfidf+feat | 0.830 ± 0.004 | 0.927 ± 0.004 | 0.876 ± 0.003 | 40.5 ± 1.3 | 15.6 ± 0.8 | 0.830 ± 0.004 | 0.927 ± 0.004 | 40.5 ± 1.3 | 15.6 ± 0.8 |
| dense:minilm+feat | 0.833 ± 0.005 | 0.919 ± 0.012 | 0.874 ± 0.007 | 39.3 ± 1.1 | 17.3 ± 2.5 | 0.833 ± 0.005 | 0.919 ± 0.012 | 39.3 ± 1.1 | 17.3 ± 2.5 |
| bm25 | 0.829 ± 0.004 | 0.918 ± 0.004 | 0.871 ± 0.002 | 40.5 ± 1.1 | 17.4 ± 0.8 | 0.829 ± 0.004 | 0.918 ± 0.004 | 40.5 ± 1.1 | 17.4 ± 0.8 |
| tfidf | 0.817 ± 0.001 | 0.923 ± 0.006 | 0.867 ± 0.003 | 44.0 ± 0.0 | 16.4 ± 1.1 | 0.817 ± 0.001 | 0.923 ± 0.006 | 44.0 ± 0.0 | 16.4 ± 1.1 |
| dense:minilm | 0.821 ± 0.002 | 0.919 ± 0.003 | 0.867 ± 0.002 | 42.8 ± 0.4 | 17.2 ± 0.6 | 0.821 ± 0.002 | 0.919 ± 0.003 | 42.8 ± 0.4 | 17.2 ± 0.6 |
| regex | 0.969 ± 0.000 | 0.582 ± 0.000 | 0.727 ± 0.000 | 4.0 ± 0.0 | 89.0 ± 0.0 | 0.969 ± 0.000 | 0.582 ± 0.000 | 4.0 ± 0.0 | 89.0 ± 0.0 |

Paired on identical folds (difference tfidf+feat minus other, negative FP = tfidf+feat better):

| pair | real FP diff mean ± sd | tfidf+feat better / tie / other better | F1 diff mean ± sd |
|---|---|---|---|
| tfidf+feat vs bm25+feat | 2.58 ± 1.66 | 1 / 0 / 11 | -0.0 ± 0.0 |
| tfidf+feat vs dense:minilm+feat | 1.17 ± 1.34 | 1 / 3 / 8 | 0.0 ± 0.01 |
