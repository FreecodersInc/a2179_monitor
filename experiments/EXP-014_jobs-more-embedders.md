# EXP-014: эмбеддеры кроме MiniLM на вакансиях (T-029, H-31)

| поле | значение |
|---|---|
| статус | planned (записано до прогона) |
| дата | 2026-11-20 |
| автор | Grok Ebanutyi, ревью Лёня |
| код (commit) | `eval/nested_cv.py` (без изменений), скоры с раннера после 327139c |
| данные | `eval/de_jobs/labeled.jsonl` 462 строки, 213 позитивов, синтетики нет. Скоры `results/<base 327139c>/targets/de_jobs/dense_scores*.jsonl` (cached-small: e5-small, e5-base и варианты, bge-base-en, mpnet, MiniLM) |
| артефакты | `reports/exp014/` |

## Гипотеза (до прогона)
**H-31 на вакансиях**: ни один эмбеддер (`dense:X+feat`) не обгоняет feat only (0.879) на ≥ 0.01 held out F1 при рабочей точке профиля (max F1). Опровержение: хотя бы один `dense:X+feat` ≥ 0.889 (среднее по 10 повторам, те же фолды).

## Процедура
```bash
python eval/nested_cv.py --target de_jobs --dense-dir results/<base>/targets/de_jobs --repeats 10 --out reports/exp014 --name jobs_embedders
```

## Результаты
Ещё не прогонялось.
