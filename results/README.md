# results

Сюда CI ([docs/ci.md](../docs/ci.md)) коммитит по одному JSON на прогон: `<UTC дата>_<short sha>.json` (при повторе `_2`, `_3`), и рядом папку с тем же именем: `eval_report.md`, `eval_report.json`, `sweep.tables.md`, `dense_scores.jsonl` (если dense считался), `setup.log`. Руками не править, коммиты с `[skip ci]`, workflow эту папку игнорирует.

Поля (`schema: 1`):
- `generated_utc`, `ok` (все шаги прошли, с учётом fallback), `tests_ok`, `total_duration_s`, `workflow_in_sync` (активный `.github/workflows/experiments.yml` совпадает с `ci/experiments.yml`)
- `commit`: `sha`, `short`, `subject`, `ref`, `run_id`, `run_attempt`, `run_url`, `actor`
- `runner`: имя, ОС, архитектура, `python`, `torch`, `cuda_version`, `cuda_available`, `device_name`, `device_count`, версии sentence-transformers и transformers (или ошибка импорта)
- `dense`: `enabled`, `reason` (cuda, cpu или почему нет), `mode`, `hf_offline`
- `steps[]` (первый `setup_venv`, потом шаги `ci/run_experiments.py`): `name`, `cmd`, `exit_code`, `signal`, `timed_out`, `duration_s`, `ok`, `fallback`, `stderr_tail` (только у упавших)
- `metrics.eval`: набор (n, positives, negatives, by_source), `embedder`, по режимам P/R/F1, TP/FP/FN/TN, пороги, 2-fold, ошибочные id
- `metrics.sweep`: `dense_source`, `tfidf_2d`, `best_hybrid` (лучший hybrid вариант с recall 1.0), все варианты с P/R/F1, порогом, 2/5-fold, real only, ошибочными id
