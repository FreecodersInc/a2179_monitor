# Nested CV, DE jobs: 462 rows, 213 positives (213 of them real), 5 folds x 10 repeats, seed 0, F1 ties: sweep, train recall >= 0.0

Hybrid mode (regex OR gate and score >= t). t, lam and min-max chosen on the training folds only. mean ± sd over repeats of the pooled test folds.

| channel | P | R | F1 | FP | FN | real P | real R | real FP | real FN |
|---|---|---|---|---|---|---|---|---|---|
| feat only | 0.829 ± 0.000 | 0.934 ± 0.000 | 0.879 ± 0.000 | 41.0 ± 0.0 | 14.0 ± 0.0 | 0.829 ± 0.000 | 0.934 ± 0.000 | 41.0 ± 0.0 | 14.0 ± 0.0 |
| dense:minilm+feat | 0.835 ± 0.002 | 0.923 ± 0.005 | 0.876 ± 0.002 | 39.0 ± 0.8 | 16.5 ± 1.0 | 0.835 ± 0.002 | 0.923 ± 0.005 | 39.0 ± 0.8 | 16.5 ± 1.0 |
| bm25+feat | 0.833 ± 0.004 | 0.918 ± 0.006 | 0.873 ± 0.003 | 39.3 ± 1.3 | 17.5 ± 1.4 | 0.833 ± 0.004 | 0.918 ± 0.006 | 39.3 ± 1.3 | 17.5 ± 1.4 |
| tfidf+feat | 0.827 ± 0.001 | 0.923 ± 0.003 | 0.872 ± 0.002 | 41.2 ± 0.4 | 16.5 ± 0.7 | 0.827 ± 0.001 | 0.923 ± 0.003 | 41.2 ± 0.4 | 16.5 ± 0.7 |
| bm25 | 0.837 ± 0.004 | 0.901 ± 0.004 | 0.868 ± 0.002 | 37.3 ± 1.3 | 21.1 ± 0.9 | 0.837 ± 0.004 | 0.901 ± 0.004 | 37.3 ± 1.3 | 21.1 ± 0.9 |
| tfidf | 0.817 ± 0.000 | 0.923 ± 0.003 | 0.867 ± 0.002 | 44.0 ± 0.0 | 16.5 ± 0.7 | 0.817 ± 0.000 | 0.923 ± 0.003 | 44.0 ± 0.0 | 16.5 ± 0.7 |
| dense:minilm | 0.821 ± 0.002 | 0.919 ± 0.002 | 0.867 ± 0.001 | 42.7 ± 0.5 | 17.2 ± 0.4 | 0.821 ± 0.002 | 0.919 ± 0.002 | 42.7 ± 0.5 | 17.2 ± 0.4 |
| regex | 0.969 ± 0.000 | 0.582 ± 0.000 | 0.727 ± 0.000 | 4.0 ± 0.0 | 89.0 ± 0.0 | 0.969 ± 0.000 | 0.582 ± 0.000 | 4.0 ± 0.0 | 89.0 ± 0.0 |

Paired on identical folds (difference tfidf+feat minus other, negative FP = tfidf+feat better):

| pair | real FP diff mean ± sd | tfidf+feat better / tie / other better | F1 diff mean ± sd |
|---|---|---|---|
| tfidf+feat vs bm25+feat | 1.9 ± 1.14 | 0 / 2 / 8 | -0.0 ± 0.0 |
| tfidf+feat vs dense:minilm+feat | 2.2 ± 0.98 | 0 / 0 / 10 | -0.0 ± 0.0 |
