# EXP-011: batch 3, 150 unseen vacancies (150 with description)

Rules v2 on the title, Horvitz-Thompson over the pool, bootstrap 95%.

| labels | positive | P (95%) | R (95%) | labeled rows | positives in sample |
|---|---|---|---|---|---|
| title | E+S | 0.84 (0.747–0.92) | 0.236 (0.191–0.306) | 150 | 92 |
| title | E | 0.04 (0.0–0.093) | 0.125 (0.0–1.0) | 150 | 6 |
| full text | E+S | 0.44 (0.32–0.56) | 0.132 (0.094–0.178) | 150 | 64 |
| full text | E | 0.04 (0.0–0.093) | 0.125 (0.0–1.0) | 150 | 6 |

Label change title → full text: 42 of 150 flip positive/negative (0.28), 45 change grade.

| cv_score AUC | title labels | full-text labels |
|---|---|---|
| A profile stack | 0.752 | 0.656 |
| B master CV stack | 0.75 | 0.647 |
