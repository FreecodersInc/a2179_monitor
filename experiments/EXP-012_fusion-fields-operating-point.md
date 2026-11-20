# EXP-012: CC против RRF (H-14, часть 1), вес заголовка (H-16), рабочая точка из профиля (T-026)

| поле | значение |
|---|---|
| статус | planned (записано до прогона) |
| дата | 2026-11-20 |
| автор | Grok Ebanutyi, ревью Лёня |
| код (commit) | `eval/nested_cv.py --rrf`, `--title-weight`, рабочая точка `eval.operating_point` в профилях |
| данные | A2179 `eval/labeled.jsonl` 430 строк (dense из results/2026-11-19_247d60c), вакансии `eval/de_jobs/labeled.jsonl` 462 строки (MiniLM из CI) |
| артефакты | `reports/exp012/` |

## Что меряем
* **RRF**: для каждого канала X ещё канал `X+feat:rrf`, скор = 1/(60 + ранг X) + 1/(60 + ранг признаков), ранги считаются относительно train фолда, λ не подбирается. Сравнение с `X+feat` (convex combination, λ на train).
* **Вес заголовка**: BM25 и TF-IDF, где заголовок повторён w раз (w = 1 база, w = 3), классическое приближение BM25F. На вакансиях текст = заголовок + первые 2000 знаков описания из кэша, на A2179 заголовок + сниппет (сниппет есть у 21 строки).
* **Рабочая точка**: протокол порога берётся из профиля (`eval.operating_point`): A2179 recall 1.0 на train (как монитор), вакансии max F1 (ADR-0034). Числа должны совпасть с EXP-004 и EXP-009, это проверка, а не гипотеза.

## Гипотезы (до прогона)
1. **H-14 часть 1**: CC (`X+feat`) выше RRF (`X+feat:rrf`) на ≥ 0.03 F1 для tfidf и лучшего dense, на обоих кейсах. Опровержение: RRF ≥ CC на любом кейсе.
2. **H-16**: на вакансиях с описанием w = 3 снимает ≥ 20% FP у bm25 и tfidf без feat при том же протоколе. Опровержение: FP падает < 5%. На A2179 проверка вырождена (сниппет у 21 из 430), считаем, но не решаем по ней.

## Процедура
```bash
python eval/nested_cv.py --target a2179_screen --dense-dir results/2026-11-19_247d60c --repeats 12 --rrf --name a2179_rrf --out reports/exp012
python eval/nested_cv.py --target de_jobs --dense-dir results/2026-11-20_a7aafe2/targets/de_jobs --repeats 12 --rrf --name jobs_rrf --out reports/exp012
python eval/nested_cv.py --target de_jobs --dense-dir results/2026-11-20_a7aafe2/targets/de_jobs --repeats 12 --text-from-cache --title-weight 1 --name jobs_text_w1 --out reports/exp012
python eval/nested_cv.py --target de_jobs --dense-dir results/2026-11-20_a7aafe2/targets/de_jobs --repeats 12 --text-from-cache --title-weight 3 --name jobs_text_w3 --out reports/exp012
```

## Результаты
Ещё не прогонялось.
