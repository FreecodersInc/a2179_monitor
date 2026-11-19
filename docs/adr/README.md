# Architecture Decision Records

Здесь записаны решения, которые уже приняты в проекте, в формате [MADR](https://adr.github.io/madr/): контекст, варианты, что выбрали и почему, последствия. Решение не переписывается задним числом: если передумали, пишем новый ADR со статусом `accepted` и старому ставим `superseded by ADR-NNNN`.

Как добавить: скопировать [`template.md`](template.md) в `NNNN-<коротко>.md` (следующий номер), заполнить, добавить строку в индекс ниже, сослаться из коммита, задачи (`work/`) или эксперимента (`experiments/`), если решение из них выросло.

Статусы: `proposed`, `accepted`, `deprecated`, `superseded by ADR-NNNN`.

## Индекс

| ADR | решение | статус | дата | коммиты |
|---|---|---|---|---|
| [0001](0001-record-architecture-decisions.md) | записывать решения как ADR (MADR) | accepted | 2026-11-19 | вместе с experiments/ и work/ |
| [0002](0002-hybrid-recall-first-matcher.md) | фильтр = regex OR похожесть с model gate, recall first | accepted | 2026-11-19 | 5d24674 |
| [0003](0003-parallel-fetch-shared-limiter-aimd.md) | процессы + asyncio, один общий семафор на все процессы, AIMD пауза на 429 | accepted | 2026-11-19 | 05ccf2f, 922d9b3 |
| [0004](0004-tracked-cards-shelve-and-full-pagination.md) | отслеживаемые карточки в shelve под одним межпроцессным lock, поиски до последней страницы | accepted | 2026-11-19 | 90f64c9 |
| [0005](0005-embedder-subprocess-probe-and-fallback.md) | эмбеддер сначала в дочернем процессе, при падении откат на regex + TF-IDF, устройство из конфига, cpu без CUDA | accepted | 2026-11-19 | a9fb3cd, d8cc345 |
| [0006](0006-kp-zero-price-is-unknown.md) | цена 0 в JSON KP значит «цена не указана», бесплатно только при явной отметке | accepted | 2026-11-19 | d8cc345 |
| [0007](0007-embedder-off-tfidf-thresholds.md) | эмбеддер выключен в примере конфига, пороги TF-IDF 0.34/0.16 по EXP-001 | accepted | 2026-11-19 | 7bd8ef1 |
| [0008](0008-ci-on-self-hosted-runner.md) | эксперименты в CI на своём раннере (`a2179-gpu`), только push в main, результаты в `results/`, копия `ci/experiments.yml` в `.github/workflows/` | accepted | 2026-11-19 | этот коммит |
| [0009](0009-thin-workflow-wrapper.md) | workflow как тонкая обёртка без `uses:`, вся логика в `ci/run.sh` | accepted | 2026-11-19 | этот коммит |
| [0010](0010-configurable-targets.md) | цель поиска как профиль `targets/<name>.json`, A2179 по умолчанию без изменений | accepted | 2026-11-19 | этот коммит |
| [0011](0011-ci-secrets.md) | секреты CI по имени из environment experiment, значения нигде не оседают | accepted | 2026-11-19 | этот коммит |
| [0012](0012-transparent-hybrid-retrieval.md) | прозрачный гибрид без векторной БД: TF-IDF, BM25, MiniLM косинус к прототипам, min-max, convex combination по сетке w, RRF как контроль, признаки отдельным членом lam | accepted | 2026-11-19 | 7bd8ef1, этот коммит |
| [0013](0013-eval-driven-decisions.md) | решения по качеству только по eval таблицам: recall first порог, FP/FN, 5-fold, real only, ранжирующие метрики и графики | accepted | 2026-11-19 | 7bd8ef1, этот коммит |
| [0014](0014-eval-enrichment-human-review.md) | набор растёт через `eval/enrich.py`, авто метки правилами с причиной, каждую смотрит человек | accepted | 2026-11-19 | ac0e7ce, 3b678fa |
| [0015](0015-ci-config-repo-variables.md) | настройки CI переменными репы (`toJSON(vars)`), дефолты CACHE_MODEL, TG_SEND, PLOTS = 1 | accepted | 2026-11-19 | 186c47d, 3b678fa, этот коммит |
| [0016](0016-working-standards.md) | стандарты работы: гипотеза → задача → EXP → прогон → ADR, где что лежит | accepted | 2026-11-19 | этот коммит |
| [0017](0017-two-case-eval.md) | гипотезы проверяются на двух кейсах параллельно, унификация (M-001, M-003) выше прогонов | accepted | 2026-11-19 | этот коммит |
| [0018](0018-ci-embedders-cpu-one-at-a-time.md) | эмбеддеры в CI на CPU, по одной модели, batch 16, не больше 4 потоков, память между моделями освобождается | accepted | 2026-11-19 | 2cf7478 |
| [0019](0019-ci-embedders-cached-small.md) | по умолчанию A2179_CI_EMBEDDERS=cached-small (до 300M параметров), большие модели только по имени | accepted | 2026-11-19 | 2cf7478 |
| [0020](0020-exp-003-results-suspect.md) | результаты EXP-003 под подозрением: признаки и пороги подбирались на тех же 430 строках | accepted | 2026-11-19 | 634fa55 |
| [0021](0021-t-023-holdout-fresh-listings.md) | признаки и порог замораживаем коммитом, меряем только на свежих ручных реальных лотах | accepted | 2026-11-19 | c571d44 |
| [0022](0022-ci-dataset-fetch.md) | сбор сырых объявлений шагом CI на раннере, вежливо и по robots.txt, для всех экспериментов | accepted | 2026-11-19 | ещё не реализовано |
| [0023](0023-runner-timeout-absolute-workdir.md) | герметичный тест run.sh (таймаут 300 с) и абсолютный workdir раннера, не литеральный ~ | accepted | 2026-11-19 | 4c9aaef |
| [0024](0024-de-jobs-bot-labels.md) | первые 112 меток de_jobs ставит ассистент, `labeler: bot`, не независимая разметка | accepted | 2026-11-19 | этот коммит |
