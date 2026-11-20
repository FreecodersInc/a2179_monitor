# EXP-009: вакансии с новым определением позитива (SRE, DevOps, ML engineer да), Data Science отдельным вариантом

| поле | значение |
|---|---|
| статус | planned (записано до прогона) |
| дата | 2026-11-20 |
| автор | Grok Ebanutyi, ревью Лёня |
| код (commit) | профиль v2 `targets/de_jobs.json` (модель `ml_sre_infra`, расширенный `strong_part`), `eval/nested_cv.py --relabel-rx`, `eval/strata_eval.py --rules-now` |
| данные | `eval/de_jobs/labeled.jsonl` 462 строки. 6 строк перемечены C → S по новому определению (ML engineer роли, старая метка в `esci_v1`), позитивов 213. Dense MiniLM из CI 927c7e7 (прототипы не менялись, скоры валидны) |
| seed | CV 0…11, bootstrap 0 |
| артефакты | `reports/exp009/` |

## Что изменилось и почему
Лёня 2026-11-20: SRE, DevOps и ML engineer вакансии это позитив, это его опыт. Data Science попробовать как позитив отдельно. Профиль v2 расширяет gate и регекс под эти роли. **Профиль v2 писался после того, как мы видели ошибки пачки 2 (EXP-005)**, поэтому всё, что меряет сам профиль на `labeled.jsonl`, это числа на тех же данных. Уже видно до прогона: gate v2 закрывает 12 из 213 позитивов (было 67 из 207). Честная проверка профиля v2 это свежая выборка из пула, которой мы не видели (пачка 3, T-025).

## Гипотезы (до прогона)
Протокол выбора порога на этот раз заявлен заранее: max F1 на train (`--min-recall 0.0`), потому что recall 1.0 недостижим, пока gate закрывает хоть один позитив.
1. **H-31 v2**: regex v2 один held out не хуже лучшей связки +feat больше чем на 0.03 F1. Опровержение: лучшая embedder+feat связка выше regex на ≥ 0.05 F1 и меньше FP, чем tfidf+feat, в ≥ 10 из 12 повторов.
2. **H-15 v2**: убрать признаки стоит tfidf ≥ 0.05 F1. Опровержение: < 0.02.
3. **Вариант DS** (Data Science, applied scientist тоже позитив, `--relabel-rx`): лучший канал меняет F1 меньше чем на 0.02 относительно базового варианта. Опровержение: ≥ 0.05 в любую сторону.
4. **H-33 повторно на правилах v2** (Horvitz-Thompson по стратам пачки 2, только правила, без модели): P ≥ 0.70 и R по E+S ≥ 0.50. Это на тех же данных, по которым писался v2, поэтому даже «держится» тут не доказательство. Опровержение: верхняя граница R < 0.50.

## Процедура
```bash
python eval/nested_cv.py --target de_jobs --dense-dir results/2026-11-19_927c7e7/targets/de_jobs --repeats 12 --min-recall 0.0 --name v2_base --out reports/exp009
python eval/nested_cv.py --target de_jobs --dense-dir results/2026-11-19_927c7e7/targets/de_jobs --repeats 12 --min-recall 0.0 --relabel-rx "data\s*scien|\bscientists?\b" --name v2_ds --out reports/exp009
python eval/strata_eval.py --rules-now --out reports/exp009
```

## Результаты
Ещё не прогонялось.
