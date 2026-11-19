# EXP-003: какие эмбеддеры тянут сербский, сравнение в один прогон

| поле | значение |
|---|---|
| статус | planned (код готов, ждём модели в кэше у Лёни) |
| дата | 2026-11-19 (запись), прогона ещё не было |
| код (commit) | этот коммит (`eval/embedders.json`, `eval/sweep.py --embedders`, bootstrap в `eval/plots.py`) |
| данные | `eval/labeled.jsonl` blob 89e2f32dfd8b, 430 строк (48 позитивов), записать blob заново, если набор вырос |
| окружение | раннер Лёни: GeForce MX130 (2 GB), torch 2.14.1+cu126, Python 3.14. Модели больше 2 GB в fp32 считаются на CPU |
| seed | bootstrap seed 0, 1000 ресемплов. Остальное детерминировано |
| артефакты | `results/<base>/dense_scores_<short>.jsonl`, таблица «Dense models side by side» в `sweep.tables.md` и в `plots.md` |

## Вопрос и гипотеза
H-30 дословно: «Мини не тянет сербский»: мультиязычные модели с сербским в обучении бьют MiniLM. Лучшая даёт dense pure P@R1 и P@R95 хотя бы на 0.10 выше MiniLM, 95% bootstrap интервалы не пересекаются, и в 5-fold выигрыш держится.
Опровержение: ни одна модель не выше MiniLM больше чем на 0.05 по P@R1 или интервалы пересекаются у всех. Тогда дело в постановке (H-01), а не в языке.
Попутно: H-17 (что дают префиксы E5: `e5-base` симметрично `query: `, `e5-base-qp` листинг query и прототипы passage, `e5-base-noprefix`).

## Условия и установка
Модели из `eval/embedders.json` (id, префиксы, размер, лицензия проверены на huggingface.co 2026-11-19). Порядок загрузки для 2 GB карты:
1. `intfloat/multilingual-e5-base` 278M, 1.1 GB, префикс `query: ` / `passage: `, влезает в MX130
2. `BAAI/bge-m3` около 568M, 2.3 GB (fp32 .bin), без префиксов, на CPU
3. `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` 278M, 1.1 GB, без префиксов, влезает
4. `intfloat/multilingual-e5-small` 118M, 0.47 GB, префиксы как у base, влезает
По желанию: `intfloat/multilingual-e5-large` (560M, 2.2 GB, CPU), `intfloat/multilingual-e5-large-instruct` (560M, 1.1 GB fp16 на диске, инструкция на стороне запроса, CPU), `sentence-transformers/LaBSE` (471M, 1.9 GB, контроль), `djovak/embedic-base` (сербский дообученный multilingual-e5-base, gated: принять условия на странице и `huggingface-cli login`).
Не берём: `jinaai/jina-embeddings-v3` (cc-by-nc-4.0 и remote code), `Alibaba-NLP/gte-multilingual-base` (код модели из другого репо через trust_remote_code), `classla/bcms-bertic`, `classla/xlm-r-bertic`, `jerteh/Jerteh-355` (fill-mask энкодеры, не модели предложений).

## Процедура
```bash
# один раз, кэш в HF_HOME (по умолчанию ~/.cache/huggingface), только нужные файлы:
python scripts/download_embedders.py multilingual english
# руками, все модели из кэша в одном прогоне:
python eval/sweep.py --compute-dense --embedders cached --plots reports/exp003_plots
```
В CI с 2cf7478 по умолчанию `A2179_CI_EMBEDDERS=cached-small` на CPU (модели до 300M, большие только по имени, см. docs/ci.md). Раньше было `cached`, считаются все модели реестра, которые лежат в кэше раннера. Модель, которой нет или которая упала, пропускается с причиной (`embedders_skipped`), остальные считаются.

## Результаты
Прогон CI 2cf7478 (CPU, 430 строк A2179, 48 позитивов): [results/2026-11-19_2cf7478.json](../results/2026-11-19_2cf7478.json), таблицы [sweep.tables.md](../results/2026-11-19_2cf7478/sweep.tables.md). Считались minilm (модель монитора), mpnet, e5-small, e5-base (три варианта префиксов), bge-base-en, e5-base-en. bge-m3, e5-large, LaBSE не считались (CPU по умолчанию, большие только по имени).

Recall 1.0, лучший порог, режим hybrid (regex ИЛИ канал):

| канал | P | F1 | FP | 5-fold F1 | AP | P@R1 |
|---|---|---|---|---|---|---|
| regex один (R 0.958, 2 FN) | 0.939 | 0.948 | 3 | | | |
| tfidf+feat, без ML | 0.857 | 0.923 | 8 | | 0.988 | 0.857 |
| feat один, без ML | | | | | 0.989 | 0.857 |
| mpnet+feat, лучший с ML | 0.842 | 0.914 | 9 | 0.902 | 0.988 | 0.842 |
| e5-base-qp+feat | 0.828 | 0.906 | 10 | 0.893 | 0.971 | 0.828 |
| minilm+feat (монитор сегодня) | 0.787 | 0.881 | 13 | 0.885 | 0.972 | 0.787 |
| minilm один | 0.640 | 0.780 | 27 | 0.793 | 0.967 | 0.640 |
| e5-small один | 0.302 | 0.464 | 111 | 0.497 | 0.959 | 0.302 |

Чистый канал (без regex) у всех эмбеддеров хуже явных признаков: AP minilm 0.363, mpnet 0.410, лучший e5-small+feat 0.796 против feat 0.767 и tfidf+feat 0.720. Префиксы e5: асимметричный `passage:`/`query:` лучше симметричного в hybrid (F1 0.706 против 0.519), H-17 в эту сторону.

Кейс de_jobs: метрик нет, набора ещё нет (шаг `eval[de_jobs]` skipped). Пул за день: 112 вакансий (Remotive 15, Remote OK 97), из них Data Engineer в заголовке 2. См. T-019.

## Выводы
На A2179 эмбеддер не нужен. Лучшая связка без ML (tfidf+feat) не хуже лучшей с ML (mpnet+feat) на тех же 430 строках: 8 FP против 9, разница в одну строку, в шуме. Модель монитора сегодня (minilm) самая слабая из всех по AP. H-30 (есть модель заметно лучше MiniLM): да, mpnet лучше minilm, но H-31 (база без ML не хуже) тоже да, и это решает: гонять GPU ради этой задачи незачем. Предложение для ADR: в мониторе A2179 по умолчанию regex + tfidf+feat, эмбеддер выключен, остаётся как опция.

## Угрозы валидности
Кириллицы в наборе 4 строки из 430, выводы про sr_cyr без новых строк не делаем. Прототипы написаны под MiniLM и TF-IDF, модель с другими префиксами может проигрывать из за формулировок (поэтому три варианта префиксов у e5-base). Bootstrap по строкам не учитывает зависимость объявлений одного продавца.

## Следующие шаги
По итогам: H-30 и H-17 в `HYPOTHESES.md`, при выигрыше ADR на смену модели монитора (заменит ADR-0007), задача T-016.
