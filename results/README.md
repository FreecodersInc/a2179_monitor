# results

Сюда CI ([docs/ci.md](../docs/ci.md)) коммитит по одному JSON на прогон: `<UTC дата>_<short sha>.json` (при повторе `_2`, `_3`), и рядом папку с тем же именем: `eval_report.md`, `eval_report.json`, `sweep.tables.md`, `dense_scores.jsonl` (если dense считался), `setup.log`, графики `eval/plots.py`: `01_dataset.png` … `09_errors_by_kind.png`, `plots.md` (метрики ранжирования + все графики с подписями), `plots.json`. Графики ещё и вклеены в конец `sweep.tables.md`. Руками не править, коммиты с `[skip ci]`, workflow эту папку игнорирует.

Поля (`schema: 3`, в 1 не было `secrets` и `skipped`, во 2 не было `metrics.ranking` и `advisory`):
- `generated_utc`, `ok` (все шаги прошли, с учётом fallback), `tests_ok`, `total_duration_s`, `workflow_in_sync` (активный `.github/workflows/experiments.yml` совпадает с `ci/experiments.yml`)
- `commit`: `sha`, `short`, `subject`, `ref`, `run_id`, `run_attempt`, `run_url`, `actor`
- `runner`: имя, ОС, архитектура, `python`, `torch`, `cuda_version`, `cuda_available`, `device_name`, `device_count`, версии sentence-transformers и transformers (или ошибка импорта)
- `secrets`: только `present` / `absent` по имени (`TG_BOT_TOKEN`, `TG_CHAT_ID`, `GITHUB_TOKEN`, ключи `A2179_CI_SECRETS`), значения никогда не пишутся и вычищаются из всего файла
- `dense`: `enabled`, `reason` (cuda, cpu или почему нет), `mode`, `hf_offline`
- `steps[]` (первый `setup_venv`, потом шаги `ci/run_experiments.py`): `name`, `cmd`, `exit_code`, `signal`, `timed_out`, `duration_s`, `ok`, `fallback`, `skipped` (например `tg_integration` без секретов), `advisory` (шаг `plots`: падение видно, но `ok` не краснеет), `stderr_tail` (только у упавших)
- `metrics.eval`: набор (n, positives, negatives, by_source), `embedder`, по режимам P/R/F1, TP/FP/FN/TN, пороги, 2-fold, ошибочные id
- `metrics.sweep`: `dense_source`, `tfidf_2d`, `best_hybrid` (лучший hybrid вариант с recall 1.0), все варианты с P/R/F1, порогом, 2/5-fold, real only, ошибочными id
- `metrics.ranking`: по каналу и порядку (pure, hybrid) R@npos, R@2npos, P@R95, P@R1, depth@R1, AP, nDCG@npos, MRR (см. `eval/plots.py`)
