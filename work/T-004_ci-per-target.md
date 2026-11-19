# T-004: CI по всем целям

| поле | значение |
|---|---|
| статус | review |
| майлстоун | M-001 |
| создана | 2026-11-19 |

## Цель
`ci/run_experiments.py` гоняет eval и свип для каждого профиля из `targets/` (или из переменной `A2179_CI_TARGETS`), результат содержит метрики по цели.

## Критерии приёмки
- [x] `metrics.targets.<target>.eval` и `.sweep` в `results/*.json` (schema 3, A2179 остаётся в `metrics.eval` / `.sweep` как раньше), отчёты в `results/<base>/targets/<target>/`
- [x] без правки YAML (только `ci/run_experiments.py` и `ci/write_results.py`)
- [x] тест на два профиля (`tests/test_target_eval.py`)

## Ссылки
- решения: ADR-0009, ADR-0010

## Заметки и итог
CI уже пишет отчёты ручной проверки по обоим публичным профилям (`eval/review_run.py`, шаг `review`). Теперь шаги `eval[<target>]` и `sweep[<target>]` идут для каждой цели из `A2179_CI_TARGETS` кроме A2179 (advisory, новая цель не красит прогон). Пока у `de_jobs` нет `eval/de_jobs/labeled.jsonl`, шаг пишется как skipped с причиной, в `metrics.targets.de_jobs.skipped`. Как только набор T-019 появится, числа по вакансиям пойдут сами, без правки CI.
