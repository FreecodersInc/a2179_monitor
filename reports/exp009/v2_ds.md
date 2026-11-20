# Nested CV, DE jobs: 462 rows, 231 positives (231 of them real), 5 folds x 12 repeats, seed 0, F1 ties: sweep, train recall >= 0.0

Hybrid mode (regex OR gate and score >= t). t, lam and min-max chosen on the training folds only. mean ± sd over repeats of the pooled test folds.

| channel | P | R | F1 | FP | FN | real P | real R | real FP | real FN |
|---|---|---|---|---|---|---|---|---|---|
| feat only | 0.884 ± 0.003 | 0.922 ± 0.000 | 0.903 ± 0.001 | 27.8 ± 1.0 | 18.0 ± 0.0 | 0.884 ± 0.003 | 0.922 ± 0.000 | 27.8 ± 1.0 | 18.0 ± 0.0 |
| bm25 | 0.891 ± 0.005 | 0.912 ± 0.002 | 0.902 ± 0.003 | 25.8 ± 1.5 | 20.2 ± 0.6 | 0.891 ± 0.005 | 0.912 ± 0.002 | 25.8 ± 1.5 | 20.2 ± 0.6 |
| bm25+feat | 0.893 ± 0.005 | 0.910 ± 0.004 | 0.902 ± 0.004 | 25.1 ± 1.3 | 20.7 ± 0.8 | 0.893 ± 0.005 | 0.910 ± 0.004 | 25.1 ± 1.3 | 20.7 ± 0.8 |
| dense:minilm+feat | 0.891 ± 0.005 | 0.910 ± 0.006 | 0.901 ± 0.004 | 25.7 ± 1.2 | 20.8 ± 1.3 | 0.891 ± 0.005 | 0.910 ± 0.006 | 25.7 ± 1.2 | 20.8 ± 1.3 |
| tfidf+feat | 0.884 ± 0.003 | 0.912 ± 0.004 | 0.898 ± 0.002 | 27.6 ± 0.8 | 20.2 ± 1.0 | 0.884 ± 0.003 | 0.912 ± 0.004 | 27.6 ± 0.8 | 20.2 ± 1.0 |
| dense:minilm | 0.886 ± 0.003 | 0.911 ± 0.002 | 0.898 ± 0.002 | 27.2 ± 0.7 | 20.5 ± 0.5 | 0.886 ± 0.003 | 0.911 ± 0.002 | 27.2 ± 0.7 | 20.5 ± 0.5 |
| tfidf | 0.882 ± 0.002 | 0.913 ± 0.006 | 0.897 ± 0.002 | 28.3 ± 0.6 | 20.2 ± 1.3 | 0.882 ± 0.002 | 0.913 ± 0.006 | 28.3 ± 0.6 | 20.2 ± 1.3 |
| regex | 0.969 ± 0.000 | 0.537 ± 0.000 | 0.691 ± 0.000 | 4.0 ± 0.0 | 107.0 ± 0.0 | 0.969 ± 0.000 | 0.537 ± 0.000 | 4.0 ± 0.0 | 107.0 ± 0.0 |

Paired on identical folds (difference tfidf+feat minus other, negative FP = tfidf+feat better):

| pair | real FP diff mean ± sd | tfidf+feat better / tie / other better | F1 diff mean ± sd |
|---|---|---|---|
| tfidf+feat vs bm25+feat | 2.5 ± 1.04 | 0 / 1 / 11 | -0.0 ± 0.0 |
| tfidf+feat vs dense:minilm+feat | 1.92 ± 1.38 | 0 / 3 / 9 | -0.0 ± 0.0 |
