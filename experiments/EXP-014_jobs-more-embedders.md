# EXP-014: эмбеддеры кроме MiniLM на вакансиях (T-029, H-31)

| поле | значение |
|---|---|
| статус | done (план записан до прогона) |
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
Скоры `results/2026-11-20_327139c/targets/de_jobs/`, таблица `reports/exp014/jobs_embedders.md`.

| канал | F1 | FP | FN |
|---|---|---|---|
| dense:bge-base-en+feat | 0.881 ± 0.006 | 29.5 | 22.2 |
| dense:bge-base-en | 0.880 ± 0.009 | 30.8 | 21.5 |
| feat only | 0.879 | 41.0 | 14.0 |
| dense:e5-base-qp+feat | 0.879 ± 0.009 | 37.0 | 16.9 |
| dense:minilm+feat | 0.876 ± 0.002 | 39.0 | 16.5 |
| dense:mpnet+feat | 0.874 ± 0.004 | 38.4 | 17.9 |

**H-31 на вакансиях: не опровергнута.** Лучший эмбеддер bge-base-en+feat 0.881, порог опровержения 0.889. Эмбеддеры по F1 не добавляют ничего к признакам профиля. Разведочно: bge меняет баланс, FP 41 → 29.5 за счёт FN 14 → 22, это может пригодиться, если рабочая точка будет про точность.

