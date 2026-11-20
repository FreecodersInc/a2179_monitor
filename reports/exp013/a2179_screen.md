# EXP-013, A2179: 430 rows, 48 positives, 10 halves + full set, 5 folds x 5 repeats, train recall >= 1.0

| channel | prototypes | P | R | F1 | FP | FN |
|---|---|---|---|---|---|---|
| bm25 | full | 0.579 | 0.979 | 0.728 ± 0.010 | 34.2 | 1.0 |
| bm25 | selected | 0.850 | 0.962 | 0.903 ± 0.009 | 8.2 | 1.8 |
| bm25+feat | full | 0.761 | 0.979 | 0.856 ± 0.009 | 14.8 | 1.0 |
| bm25+feat | selected | 0.862 | 0.962 | 0.910 ± 0.006 | 7.4 | 1.8 |
| tfidf | full | 0.671 | 0.979 | 0.797 ± 0.007 | 23.0 | 1.0 |
| tfidf | selected | 0.751 | 0.979 | 0.850 ± 0.011 | 15.6 | 1.0 |
| tfidf+feat | full | 0.848 | 0.979 | 0.909 ± 0.007 | 8.4 | 1.0 |
| tfidf+feat | selected | 0.837 | 0.979 | 0.902 ± 0.010 | 9.2 | 1.0 |

Chosen on the training folds (count over folds x repeats):

* bm25: half3 15, half1 5, half8 5
* bm25+feat: half3 13, half8 7, half1 5
* tfidf: half1 20, half4 5
* tfidf+feat: half7 13, full 6, half2 4, half5 1, half0 1
