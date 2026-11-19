# results

Сюда CI ([docs/ci.md](../docs/ci.md)) коммитит по одному JSON на прогон: `<UTC дата>_<short sha>.json` (при повторе `_2`, `_3`). Руками не править, коммиты с `[skip ci]`, workflow эту папку игнорирует.

Поля (`schema: 1`):
- `generated_utc`, `ok` (все шаги прошли, с учётом fallback), `tests_ok`, `total_duration_s`
- `commit`: `sha`, `short`, `subject`, `ref`, `run_id`, `run_attempt`, `run_url`, `actor`
- `runner`: имя, ОС, архитектура, `python`, `torch`, `cuda_version`, `cuda_available`, `device_name`, `device_count`, версии sentence-transformers и transformers (или ошибка импорта)
- `dense`: `enabled`, `reason` (cuda, cpu или почему нет), `mode`, `hf_offline`
- `steps[]`: `name`, `cmd`, `exit_code`, `signal`, `timed_out`, `duration_s`, `ok`, `fallback`, `stderr_tail` (только у упавших)
- `metrics.eval`: набор (n, positives, negatives, by_source), `embedder`, по режимам P/R/F1, TP/FP/FN/TN, пороги, 2-fold, ошибочные id
- `metrics.sweep`: `dense_source`, `tfidf_2d`, `best_hybrid` (лучший hybrid вариант с recall 1.0), все варианты с P/R/F1, порогом, 2/5-fold, real only, ошибочными id
