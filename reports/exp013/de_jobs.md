# EXP-013, DE jobs: 462 rows, 213 positives, 10 halves + full set, 5 folds x 5 repeats, train recall >= 0.0

| channel | prototypes | P | R | F1 | FP | FN |
|---|---|---|---|---|---|---|
| bm25 | full | 0.827 | 0.920 | 0.871 ± 0.002 | 41.2 | 17.0 |
| bm25 | selected | 0.842 | 0.913 | 0.876 ± 0.006 | 36.6 | 18.4 |
| bm25+feat | full | 0.837 | 0.923 | 0.878 ± 0.003 | 38.2 | 16.4 |
| bm25+feat | selected | 0.849 | 0.917 | 0.882 ± 0.005 | 34.8 | 17.6 |
| tfidf | full | 0.817 | 0.921 | 0.866 ± 0.003 | 44.0 | 16.8 |
| tfidf | selected | 0.842 | 0.920 | 0.880 ± 0.005 | 36.6 | 17.0 |
| tfidf+feat | full | 0.831 | 0.927 | 0.876 ± 0.003 | 40.2 | 15.6 |
| tfidf+feat | selected | 0.844 | 0.919 | 0.880 ± 0.003 | 36.2 | 17.2 |

Chosen on the training folds (count over folds x repeats):

* bm25: half3 17, half2 5, half7 2, full 1
* bm25+feat: half7 11, half3 9, half2 4, full 1
* tfidf: half7 24, half6 1
* tfidf+feat: half7 23, half5 2
