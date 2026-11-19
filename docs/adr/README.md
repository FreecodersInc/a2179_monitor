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
