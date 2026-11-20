# Nested CV, DE jobs: 462 rows, 207 positives (207 of them real), 5 folds x 12 repeats, seed 0, F1 ties: sweep, train recall >= 0.9

Hybrid mode (regex OR gate and score >= t). t, lam and min-max chosen on the training folds only. mean ± sd over repeats of the pooled test folds.

| channel | P | R | F1 | FP | FN | real P | real R | real FP | real FN |
|---|---|---|---|---|---|---|---|---|---|
| feat only | 0.448 ± 0.000 | 1.000 ± 0.000 | 0.619 ± 0.000 | 255.0 ± 0.0 | 0.0 ± 0.0 | 0.448 ± 0.000 | 1.000 ± 0.000 | 255.0 ± 0.0 | 0.0 ± 0.0 |
| tfidf | 0.448 ± 0.000 | 1.000 ± 0.000 | 0.619 ± 0.000 | 255.0 ± 0.0 | 0.0 ± 0.0 | 0.448 ± 0.000 | 1.000 ± 0.000 | 255.0 ± 0.0 | 0.0 ± 0.0 |
| tfidf+feat | 0.448 ± 0.000 | 1.000 ± 0.000 | 0.619 ± 0.000 | 255.0 ± 0.0 | 0.0 ± 0.0 | 0.448 ± 0.000 | 1.000 ± 0.000 | 255.0 ± 0.0 | 0.0 ± 0.0 |
| bm25 | 0.448 ± 0.000 | 1.000 ± 0.000 | 0.619 ± 0.000 | 255.0 ± 0.0 | 0.0 ± 0.0 | 0.448 ± 0.000 | 1.000 ± 0.000 | 255.0 ± 0.0 | 0.0 ± 0.0 |
| bm25+feat | 0.448 ± 0.000 | 1.000 ± 0.000 | 0.619 ± 0.000 | 255.0 ± 0.0 | 0.0 ± 0.0 | 0.448 ± 0.000 | 1.000 ± 0.000 | 255.0 ± 0.0 | 0.0 ± 0.0 |
| dense:minilm | 0.448 ± 0.000 | 1.000 ± 0.000 | 0.619 ± 0.000 | 255.0 ± 0.0 | 0.0 ± 0.0 | 0.448 ± 0.000 | 1.000 ± 0.000 | 255.0 ± 0.0 | 0.0 ± 0.0 |
| dense:minilm+feat | 0.448 ± 0.000 | 1.000 ± 0.000 | 0.619 ± 0.000 | 255.0 ± 0.0 | 0.0 ± 0.0 | 0.448 ± 0.000 | 1.000 ± 0.000 | 255.0 ± 0.0 | 0.0 ± 0.0 |
| regex | 0.978 ± 0.000 | 0.435 ± 0.000 | 0.602 ± 0.000 | 2.0 ± 0.0 | 117.0 ± 0.0 | 0.978 ± 0.000 | 0.435 ± 0.000 | 2.0 ± 0.0 | 117.0 ± 0.0 |

Paired on identical folds (difference tfidf+feat minus other, negative FP = tfidf+feat better):

| pair | real FP diff mean ± sd | tfidf+feat better / tie / other better | F1 diff mean ± sd |
|---|---|---|---|
| tfidf+feat vs bm25+feat | 0 ± 0.0 | 0 / 12 / 0 | 0.0 ± 0.0 |
| tfidf+feat vs dense:minilm+feat | 0 ± 0.0 | 0 / 12 / 0 | 0.0 ± 0.0 |
