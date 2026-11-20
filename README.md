# a2179_monitor: стенд для исследования сопоставления на двух кейсах

Вопрос исследования: **может ли простой гибрид (правила профиля, лексика, эмбеддер) надёжно решать «это то самое, а не про то же», и где ML реально добавляет что-то к правилам.** Ответ проверяется на двух кейсах одним кодом и одним протоколом ([ADR-0017](docs/adr/0017-two-case-eval.md)).

1. **Экран A2179** (`targets/a2179_screen.json`). Простой кейс, который проверяется глазами: подходит ли объявление на сербских досках как экран или крышка в сборе для MacBook Air 13" 2020 Intel A2179. Каждую ошибку видно в списке строк.
2. **Расстояние CV до вакансии** (`targets/de_jobs.json`, `cv/master_cv.md`). Подтверждающий кейс: насколько вакансия близка к опыту кандидата (Data Engineer, ML platform, SRE). Тут глазами не проверить, нужны метки и ранжирование.

Если вывод держится на обоих кейсах, он про метод, а не про один набор данных.

## Честные числа на 2026-11-20 (плохие новости первыми)
* **Экран A2179, свежая пачка: recall не измерен.** CI собрал 13 новых лотов после заморозки настроек, позитивов среди них 0, ложных срабатываний 0. Полноту на свежих данных этим не проверить.
* **Набор A2179 частично ручной.** `eval/labeled.jsonl` это 430 строк, из них 59 синтетических. Из 48 позитивов 31 синтетический, реальных 17. Исторические числа EXP-004 и EXP-013 ниже по нему, поэтому оптимистичны. EXP-021, EXP-024, EXP-026 отдельно используют только реальные строки.
* **Вакансии, свежая пачка 3 (EXP-011), прочитана целиком: главные результаты не переносятся.** Правила v2 на 150 невиденных вакансиях: R по E+S 0.236 по меткам по заголовку и 0.132 по меткам по тексту, P по тексту 0.44. Скор близости к CV на метках по полному тексту AUC 0.656 (профиль) и 0.647 (полный CV), против 0.902 и 0.904 на старом наборе. 28% меток меняются, когда читаешь описание.
* **Все метки ставил один разметчик** (бот, ревью Лёни выборочно). Второго нет (H-07).
* A2179, вложенная CV на 430 строках: regex профиля один F1 0.948, лучший гибрид tfidf+feat 0.907, лучший эмбеддер mpnet+feat 0.904, MiniLM+feat 0.872. ML пока не бьёт правила (EXP-004, EXP-012).
* Вакансии, вложенная CV на 462 строках (213 позитивов, синтетики нет): признаки профиля без ML F1 0.879, MiniLM+feat 0.876, regex v2 один 0.727 (EXP-009). Скор по опыту: стек из профиля AUC 0.773, из полного CV 0.805 (EXP-010).
* Эмбеддеры кроме MiniLM на вакансиях (e5, bge, mpnet): лучший bge-base-en+feat 0.881 против признаков 0.879, ML не добавляет (EXP-014).
* RRF не хуже convex combination: A2179 tfidf 0.915 против 0.907, вакансии ±0.002 (EXP-012, H-14 опровергнута).
* Отбор прототипов BM25 внутри train: A2179 bm25 0.728 → 0.903 F1, FP 34 → 8 (EXP-013, H-39). На свежих данных не проверено.

* H-43, свежие 150: TinyBERT по заголовку AUC 0.787 против cv_score 0.608, фьюжн рангов 0.748 (EXP-022). Проверялся cv_score A, а приложение смешивает B плюс признаки. Это основание пробовать реранк, не доказательство прироста качества приложения.
* H-45 refuted: сетка k1 и b на 371 реальной строке A2179 даёт F1 0.686 → 0.686, вакансии 0.770 → 0.769. Параметры не меняем (EXP-026).
* KP daily 2026-11-20: 200 лотов, 18 новых, 0 позитивов. Для H-38 по прежнему 12 из нужных 15 позитивов с карточкой.

Статусы и решения: [`work/STATUS.md`](work/STATUS.md), [`experiments/HYPOTHESES.md`](experiments/HYPOTHESES.md), [`reports/PROJECT_REPORT.md`](reports/PROJECT_REPORT.md). `eval/status_check.py --check` сверяет их в тестах CI.

## Мнение: работает ли
* **Экран классифицируется правильно? Доказано частично.** На старом наборе regex и гибрид почти без ошибок, но набор наполовину по позитивам синтетический, а на свежей пачке не было ни одного позитива. Нужна свежая пачка с позитивами.
* **Мониторинг вакансий по CV работает как прототип ранжирования, но на реальных данных не доказан.** На старом наборе AUC 0.90, на свежих вакансиях, прочитанных целиком, 0.65. Слова стека в тексте почти не отличают «строит платформу» от «обучает модели ранжирования».
* **Что нужно:** реальные размеченные свежие вакансии с полным текстом (старые 462 уже переразмечены по тексту, EXP-020), пачка с большим числом позитивов (обогащённая data engineer), рабочая точка (порог) на свежих данных и живой пробный прогон монитора по вакансиям с ручной проверкой.

## Структура
| папка | что там |
|---|---|
| [`work/`](work/README.md) | майлстоуны и задачи T-xxx, таблица того, что стоит на данных |
| [`experiments/HYPOTHESES.md`](experiments/HYPOTHESES.md) | гипотезы H-xx с критериями опровержения и статусом, ссылки arXiv или «no direct arXiv support found» |
| [`docs/adr/`](docs/adr/README.md) | решения (ADR, MADR) |
| [`experiments/`](experiments/README.md) | EXP-xxx: план до прогона, процедура, результаты |
| `reports/` | таблицы и графики экспериментов, [`reports/PROJECT_REPORT.md`](reports/PROJECT_REPORT.md) сводный отчёт |
| [`results/`](results/README.md) | то, что CI коммитит после каждого push в main |
| `ci/` | логика CI: `run.sh` → `select_steps.py` (skip, targeted, full) → `run_experiments.py`, фетчер свежих лотов, `write_results.py` |
| `eval/` | оценка: `run_eval.py`, `sweep.py`, `nested_cv.py`, `freeze.py`, `cv_distance.py`, `batch3.py`, `proto_select.py`, `strata_eval.py` |
| `targets/` | профили целей: правила, прототипы, признаки, `eval.operating_point` |
| `cv/master_cv.md` | полное CV кандидата для кейса 2 (опубликовано с согласия, ADR-0035) |

Протокол: гипотеза и критерий опровержения записываются и коммитятся до прогона. Пороги, λ и min-max выбираются только на train фолдах (вложенная CV, grouped folds по заголовку). Свежие пачки не участвуют в подборе. Синтетика в свежие пачки не добавляется ([ADR-0031](docs/adr/)).

### CI и фетчер
Тонкий workflow, вся логика в `ci/` ([docs/ci.md](docs/ci.md)). Каждый push в main: селектор решает, какие тесты и шаги нужны, раннер считает eval и sweep по каждой цели, результаты коммитятся в `results/<дата>_<sha>/`. Свежая пачка A2179: закоммить `ci/requests/fetch_listings.json` с новым `id`, шаг `fetch_listings` отработает один раз и положит лоты в `results/<base>/reports/fresh_a2179/<id>/`.

### Заморозка и свежая пачка (eval/freeze)
```bash
python eval/freeze.py freeze --dense-dir results/2026-11-19_247d60c --out eval/frozen/a2179_screen.json   # до сбора
python eval/freeze.py score --frozen eval/frozen/a2179_screen.json --data eval/holdout/fresh_2026-11-20.jsonl
```
Настройки фиксируются и коммитятся до сбора пачки, на свежих строках ничего не подбирается.

## Как запустить кейс 1: экран A2179
```bash
python eval/run_eval.py                                                    # regex, tfidf, гибрид
python eval/sweep.py                                                       # каналы и слияния, отчёт по порогам
python eval/nested_cv.py --dense-dir results/2026-11-19_247d60c --rrf --out reports/exp012 --name a2179_rrf
python eval/proto_select.py --target a2179_screen --out reports/exp013
python monitor.py --dry-run --list                                         # живой прогон монитора без уведомлений
```
Глазами: `reports/<дата>/a2179_screen/matches.md` и ошибки в отчётах eval.

## Как запустить кейс 2: расстояние CV до вакансии
```bash
python eval/run_eval.py --target de_jobs
python eval/nested_cv.py --target de_jobs --dense-dir results/2026-11-20_a7aafe2/targets/de_jobs --out reports/exp012 --name jobs_rrf
python eval/cv_distance.py --stack-from cv/master_cv.md --out reports/exp010_b   # скор близости к CV
python eval/batch3.py score --out reports/exp011                                # пачка 3: метки по заголовку и по тексту
python eval/review_run.py                                                       # свежие вакансии, reports/<дата>/de_jobs/
```

## Тесты
```bash
python -m unittest discover -s tests -t .
```

## Монитор A2179: обычное использование

### Режим вакансий по CV
```bash
python monitor.py --target de_jobs --dry-run --list   # свежие вакансии по близости к cv/master_cv.md, ничего не шлёт
python monitor.py --target de_jobs --seed             # первый раз: запомнить текущие без уведомлений
python monitor.py --target de_jobs --top 10           # прислать до 10 новых верхних (Telegram, если настроен)
python monitor.py --target de_jobs --every 360 --min-score 0.6 --job-sources remotive,remoteok,ats
```
Источники: Remotive, Remote OK и публичные ATS доски (Greenhouse, Lever, Ashby). Arbeitnow, We Work Remotely RSS и HN подключены (EXP-018). Базовый скор = 0.5 · близость к CV + 0.5 · признаки профиля. Если TinyBERT уже в локальном кэше, итог = среднее нормированных рангов базового скора и cross-encoder по заголовку (ADR-0036). `--job-rerank auto` по умолчанию ничего не скачивает, `off` оставляет базовый скор, `download` разрешает загрузку только после проверки размера до 50 MiB. С реранком `--min-score` означает ранг 0..1 внутри текущей пачки, не вероятность. Уже виденные лежат в `data/jobs_seen.json`.
**Честно:** на свежих вакансиях, размеченных по полному тексту, AUC близости к CV 0.656 (EXP-011). Пока это помощь в сортировке, а не судья: верхние смотреть глазами.
Монитор объявлений: **экран (LCD панель) или крышка в сборе для MacBook Air 13" 2020 Intel A2179** в Сербии.
Источники: KupujemProdajem (поиск), ifix.rs, ihelp.rs, laptop-punjaci.com (страницы товаров и поиск по сайту). В плане: Limundo и KupindO как второй источник б/у (T-033, с dev машины оба отвечают 403 или проверкой, ждём пробу с раннера). AliExpress и Temu только справочная цена новых экранов и матриц (T-034), eBay только сравнение цены с доставкой в Сербию (T-035), оба сейчас blocked по robots.txt. OLX и Allegro не берём.
Без LLM. Фильтр гибридный: регулярки + похожесть на эталонные фразы (локальный эмбеддер HF, если установлен, иначе TF-IDF по символьным n-граммам).
Приоритет полноты: лучше лишнее уведомление, чем пропущенное объявление.

### Что ловит и что нет
- Да: A2179, EMC 3302, 661-15389, «MacBook Air 2020 Intel», латиница и кириллица (ekran, displej, lcd, panel, matrica, komplet, kućište, екран, дисплеј, экран, матрица, крышка, в сборе), доноры «za delove / u delove / на запчасти», бесплатные отдачи.
- Нет: только M1 A2337, A2338, чехлы, cover, плёнки и стёкла, клавиатуры, батареи, зарядки, шлейфы, винты, услуги «zamena ekrana», «tražim/kupujem», целые ноутбуки по характеристикам.
- **A1932 (2018/2019):** магазины продают одну и ту же панель как «A1932 | A2179» (например ifix.rs), поэтому объявления только с A1932 по умолчанию тоже ловятся с пометкой «совместимость уточни у продавца». Выключить: `"accept_a1932": false`.

### Уведомления
- 🆕 новое подходящее объявление, 📉 цена упала (по умолчанию на 3% и больше), 🎯 цена ниже цели (по умолчанию **$20**, курс RSD/EUR/USD в конфиге).
- Дубли отсекаются по локальной базе `data/state.sqlite` (история цен и уведомлений там же).
- Канал: Telegram бот (твой токен и chat id через переменные окружения), плюс консоль, `data/alerts.log` и системное уведомление (Linux `notify-send`, macOS).
- Если KupujemProdajem блокирует запросы (403) 6 запусков подряд, придёт одно предупреждение. Магазины при этом проверяются как обычно.

### Установка
```bash
cd a2179_monitor
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# по желанию, эмбеддер: сначала torch под свою ОС (см. «Эмбеддер не работает»), потом
pip install -r requirements-embed.txt
python monitor.py --check-embedder
```
Telegram: создай бота у @BotFather, напиши ему что угодно, chat id возьми из `https://api.telegram.org/bot<TOKEN>/getUpdates`.
```bash
cp .env.example .env               # впиши TG_BOT_TOKEN и TG_CHAT_ID
cp config.example.json config.json # по желанию: цель, курсы, источники, пороги
python monitor.py --test-alert     # проверка канала
```

### Запуск
```bash
python monitor.py --dry-run --list   # посмотреть, что найдётся, ничего не отправлять и не запоминать
python monitor.py --seed             # первый раз: запомнить текущие объявления без потока уведомлений
python monitor.py                    # обычный проход: найти, сравнить с базой, уведомить
python monitor.py --no-embed         # без эмбеддера, только регулярки + TF-IDF
python monitor.py --every 60         # крутиться самому раз в 60 минут (вместо cron)
python monitor.py check              # то же, что --check-embedder: диагностика эмбеддера: версии, кэш, загрузка, тестовый encode
python monitor.py --require-embedder # без эмбеддера не работать (код выхода 3), по умолчанию откат на TF-IDF
python monitor.py --workers 3 --max-concurrency 10   # параллельная загрузка (это значения по умолчанию)
python monitor.py --workers 0        # старый последовательный режим (requests, пауза между запросами)
```

#### Параллельная загрузка
Проход почти целиком состоит из ожидания сети, поэтому источники качаются и разбираются параллельно:
- `parallel.workers` (по умолчанию **3**, `--workers`, `A2179_WORKERS`) процессов, в каждом свой цикл asyncio и **одна** `aiohttp.ClientSession` на процесс. Пока одна страница разбирается, остальные запросы уже ждут ответа.
- `parallel.max_concurrency` (по умолчанию **10**, `--max-concurrency`, `A2179_MAX_CONCURRENCY`): сколько GET запросов одновременно в полёте **всего**, по всем процессам вместе, ретраи тоже считаются. Как сделано: **один общий семафор над процессами** (`multiprocessing.BoundedSemaphore`, отдаётся воркерам при старте, без отдельного процесса Manager). Слот берётся прямо перед запросом и отдаётся, как только ответ прочитан, поэтому процесс без работы слотов не держит и все слоты достаются тем, у кого работа есть. Воркеры берут слот неблокирующим опросом (раз в 5..50 мс), цикл событий не встаёт.
- Ретраи: сетевые ошибки, 5xx и 429 повторяются до `http.retries` раз (по умолчанию **3**) с экспоненциальной паузой и джиттером: `backoff_s * 2^(n-1)`, умноженное на случайное 0.5..1.5, не больше `http.backoff_max_s` (по умолчанию 5, 10, 20 с, потолок 60 с). 404 и прочие 4xx не повторяются, 403 повторяется `http.retry_403` раз, как раньше.
- 429 и лимит сайта: у каждого сайта своя пауза между стартами запросов, общая для всех процессов (AIMD). На каждый 429 пауза умножается на `delay_up` (2.0), но не меньше `delay_on_429_s` (0.5 с), а `Retry-After` в секундах останавливает старты к этому сайту на указанное время. На каждый успешный ответ пауза уменьшается на `delay_down_s` (0.05 с). Границы `delay_min_s` (0) и `delay_max_s` (30 с), стартовая пауза `delay_s` (0). Так скорость сама подходит к границе лимита сайта и держится чуть ниже неё. Выученная пауза переходит в следующий проход того же запуска (`--every`). Хочешь осторожнее: подними `delay_min_s` или `delay_on_429_s`, быстрее отходить после бана: подними `delay_down_s`.
- Результаты собираются в исходном порядке источников, поэтому дедупликация, уведомления, база и метрики те же, что в последовательном режиме. Если KP ответил блокировкой, остальные запросы к KP в этом проходе уже не стартуют (те, что уже в полёте, долетают, но в учёт не идут, как и раньше).
- Нет `aiohttp` (`pip install -r requirements.txt`): монитор пишет об этом одну строку и проходит последовательно.
- В `data/metrics.json` в `notes` видно, как прошла загрузка, например `fetch=parallel processes=3 max_concurrency=10 (one shared limiter)` и выученные паузы `pace kupujemprodajem.com=0.85s`. Счётчики `fetch.retries` и `fetch.rate_limited` считают ретраи и ответы 429, в логе событие `rate_limited` с текущей паузой.

#### Пагинация поисков
Поиски магазинов (WooCommerce) и KP читаются **до последней страницы**, пока страницы есть. Каждая страница идёт через тот же общий лимит, паузу на 429 и ретраи.
- Следующая страница: `<link rel="next">`, `<a rel="next">` или WooCommerce `a.next.page-numbers`. Для KP ещё ссылка на тот же путь с `page=N+1`, а если её нет, адрес с `page=N+1` (разметка KP вживую не проверена, параметр задаёт `pagination.kp_page_param`, пустая строка выключает догадку).
- Стоп: пустая страница (`empty`), нет ссылки на следующую (`no_next`), на странице нет новых объявлений (`no_new`), 404 или 410 после первой страницы (`end`, так WooCommerce отвечает за последней), другая ошибка (`error`), KP заблокировал (`blocked`, дальше по KP в этом проходе не идём, как и раньше), ссылка ведёт на уже прочитанную страницу (`loop`) или на другой сайт (`off_host`).
- `pagination.max_pages` (по умолчанию **500**) только аварийный потолок на один поиск (`ceiling`), низкого лимита страниц нет. `pagination.enabled: false` вернёт чтение только первой страницы.
- В метриках `pagination.shop_search.pages`, `pagination.kp.pages` и `pagination.stop.<причина>`.

#### Отслеживаемые карточки
Если в результатах поиска магазина нашлась подходящая карточка (те же регулярки и похожесть по смыслу, что для уведомлений), она попадает в постоянный список `data/tracked.shelve` и дальше каждый проход проверяется как `shop_products`: цена, наличие, статус. Так товар не теряется, даже если пропадёт из поиска. Объявления KP так не отслеживаются (парсера страницы объявления KP нет, их покрывает поиск KP со всеми страницами).
- Запросы к карточкам идут через тот же общий лимит и паузу на 429.
- Хранилище `shelve` (файл выбирает бэкенд dbm, например `tracked.shelve` или `tracked.shelve.db`). Защита от гонок: **один** `multiprocessing.Lock`, его, как и семафор запросов, главный процесс создаёт и отдаёт воркерам при старте. Любое чтение или запись берёт этот замок, открывает shelve, делает своё и закрывает его до того, как замок отпущен, так что shelve никогда не открыт дважды и запись уже на диске, когда к нему подходит следующий процесс. Воркеры пишут только результат проверки своих карточек (запись идёт в отдельном потоке, цикл событий не встаёт), новые карточки добавляет главный процесс после фильтра. `--dry-run` ничего не пишет.
- Замок защищает процессы одного запуска. Два отдельных запуска на одной папке `data` одновременно не защищены: не запускай cron, пока крутится `--every`.
- Карточка, которая `tracked.drop_after_misses` проходов подряд (по умолчанию 3) отвечает 404 или 410, помечается `gone` и больше не запрашивается. Из файла она не удаляется.
- `tracked.enabled: false` выключает список целиком, `tracked.file` меняет имя файла в `data_dir`.

#### По расписанию
Linux/macOS, `crontab -e`, раз в час:
```
17 * * * * cd /путь/к/a2179_monitor && .venv/bin/python monitor.py >> data/cron.log 2>&1
```
Windows, Планировщик заданий (PowerShell):
```powershell
schtasks /Create /SC HOURLY /TN a2179_monitor /TR "C:\путь\a2179_monitor\.venv\Scripts\python.exe C:\путь\a2179_monitor\monitor.py"
```
Чаще раза в 30 минут не надо: KP и магазины не любят частые запросы.

### Конфиг (`config.json`, все поля необязательны)
Порядок: общие defaults ← профиль цели ← `config.json` ← переменные окружения. **Цель** (что ищем: правила regex, прототипы, источники, цена, пороги) это профиль `targets/<name>.json`, по умолчанию [`targets/a2179_screen.json`](targets/a2179_screen.json), см. [ADR-0010](docs/adr/0010-configurable-targets.md). Другая цель: `"target": "my_target"` (файл `targets/my_target.json`) или путь к .json, либо `A2179_TARGET=...`.

| поле | что |
|---|---|
| `target` | профиль цели (по умолчанию `a2179_screen`) |
| `target_price` | цена цели `{"amount": 20, "currency": "USD"}` (USD, EUR, RSD), пересчёт по `rates` (ADR-0038) |
| `target_usd` | старое имя цены в USD, явное значение перекрывает цену профиля (A2179 по прежнему 20 USD) |
| `rates.rsd_per_eur`, `rates.usd_per_eur` | курсы для пересчёта |
| `price_drop_min_pct` | минимальное падение цены для 📉 |
| `alert_on` | какие события слать |
| `accept_a1932` | ловить ли объявления только с A1932 |
| `embedder.enabled` | `auto` или `true` (пробовать, при ошибке откат на TF-IDF), `false` (не трогать), `required` (без него выход с кодом 3) |
| `embedder.local_only` | `true`: не ходить в сеть, брать модель только из кэша HF |
| `embedder.model` | модель sentence-transformers, по умолчанию `paraphrase-multilingual-MiniLM-L12-v2` |
| `embedder.device` | `auto` (по умолчанию: `cuda`, если `torch.cuda.is_available()`, иначе `cpu`), `cuda`, `cuda:1`, `cpu`, `mps`. Env `A2179_DEVICE` |
| `embedder.backend` | `torch` (по умолчанию), `onnx` или `openvino` (нужен sentence-transformers 3.2+ и `optimum`), см. «Эмбеддер не работает» |
| `embedder.probe`, `embedder.probe_timeout_s` | `true`, `600`: сначала грузить модель в отдельном процессе, падение там не роняет монитор |
| `thresholds` | пороги канала похожести (подобраны eval-скриптом) |
| `sources` | URL товаров, поисков и ключевые слова KP |
| `http` | пауза между запросами к одному сайту (последовательный режим), ретраи (`retries` 3, `backoff_s`, `backoff_max_s`, `retry_403`), таймаут, User-Agent |
| `pagination` | `enabled` (true), `max_pages` (500, аварийный потолок на поиск), `kp_page_param` (`page`), см. «Пагинация поисков» |
| `tracked` | `enabled` (true), `file` (`tracked.shelve`), `drop_after_misses` (3), см. «Отслеживаемые карточки» |
| `parallel` | `workers` (3, `0` = последовательно), `max_concurrency` (10, всего по всем процессам), паузы при 429: `delay_s`, `delay_min_s`, `delay_max_s`, `delay_up`, `delay_down_s`, `delay_on_429_s`, см. «Параллельная загрузка» |

Переменные окружения: `TG_BOT_TOKEN`, `TG_CHAT_ID`, `A2179_TARGET_USD`, `A2179_RSD_PER_EUR`, `A2179_USD_PER_EUR`, `A2179_EMBEDDER`, `A2179_MODEL`, `A2179_WORKERS`, `A2179_MAX_CONCURRENCY`.

### Наблюдаемость
- `data/logs/monitor.jsonl`: структурные логи (каждый запрос, статус, задержка, решение фильтра с причинами).
- `data/tracked.shelve`: отслеживаемые карточки (см. выше).
- `data/metrics.json` и `data/metrics_history.jsonl`: число запросов, ошибки по сайтам и HTTP статусам, задержки (avg/p95/max), сколько объявлений, совпадений, событий и отправок. При `--dry-run` пишется `data/metrics_dryrun.json`.

### Пополнить eval реальными объявлениями (`eval/enrich.py`)
Запускать у себя (с сербского IP, KP режет серверные адреса). Ходит тем же кодом, что монитор: общий лимитер, пауза на хост, ретраи, пагинация. Если KP ответил 403 или капчей, KP до конца прогона больше не трогается, магазины собираются дальше.
```bash
python eval/enrich.py collect --dry-run            # только показать, какие поиски уйдут (ничего не качает)
python eval/enrich.py collect --csv                # 43 запроса: A2179/Air 2020 + детали + намерения, до 3 страниц на поиск
python eval/enrich.py collect --queries my.txt --max-pages 2 --limit 300 --csv   # свой список (строка = запрос, # комментарий)
python eval/enrich.py collect --offline-html saved_pages/ --csv                  # разобрать сохранённые страницы без сети
python eval/enrich.py collect --workers 0 --csv      # последовательно, пауза http.min_delay_s на хост: мягче для KP, если он начал резать
```
Пишет `eval/candidates_<дата>.jsonl` (и `.csv` с `--csv`). То, что уже есть в `eval/labeled.jsonl` (тот же URL или тот же заголовок), и прототипы отбрасываются. У каждой строки авто метка только по прозрачным правилам, без LLM: regex монитора, признаки товара (слова дисплея против шарниров, кулеров, клавиатуры, батареи, чехлов, стекла, кабелей, платы), намерение (`sell`, `seek`, `service`), оценка матчера (TF-IDF), поле `reason` с объяснением, `needs_review: true`, `reviewed: false`. Порядок: сначала `priority 0` (правила и матчер или признаки/намерение не согласны), потом `1` (кандидаты в трудные негативы: та же модель, слово «ekran» или высокая похожесть, но авто метка 0), потом `2` (уверенные позитивы), потом остальное.

Ревью: в CSV поставь в колонке `reviewed` 1 (или true, да, x), поправь `label` (1 = экран/крышка/донор под A2179, 0 = нет) и `kind` где надо. В JSONL то же самое полями `"reviewed": true`, `"label"`. Потом:
```bash
python eval/enrich.py merge --reviewed eval/candidates_<дата>.csv --dry-run   # посмотреть, что добавится
python eval/enrich.py merge --reviewed eval/candidates_<дата>.csv             # дописать в eval/labeled.jsonl
python eval/run_eval.py && python eval/sweep.py                              # пересчитать метрики на новом наборе
```
В `labeled.jsonl` попадают только строки с `reviewed`, с `source: "real"`, новыми id после последнего, URL и запросом. Повторный merge того же файла ничего не дублирует, старые строки не трогаются.

### Тесты
```bash
python -m unittest discover -s tests -t .
```
Тесты сеть не трогают, путь эмбеддера проверяется на поддельном модуле sentence_transformers (все ошибки выше): HTML магазинов сохранён в `tests/fixtures` (снимки 19.11.2026), KP заменён синтетической страницей.
Реальные данные: `tests/fixtures/dryrun_2026-11-19.jsonl` (17 совпадений из живого `--dry-run --list` с сербского IP: заголовок, цена, $, источник, ссылка) и `tests/fixtures/kp_next_data_dryrun.html` (12 объявлений KP из того же прогона), тесты в `tests/test_real_dryrun.py`: каждое реальное объявление должно ловиться regex и гибридом, KP разбирается с теми же ценами и городами, «0 RSD» не даёт 🎯.

### Эмбеддер не работает
Монитор без эмбеддера работает полностью (regex + TF-IDF). Если эмбеддер включён, но не поднялся, при каждом запуске будет три строки `[embedder] ...` с причиной и что делать, а в `data/logs/monitor.jsonl` событие `embedder_unavailable` с полем `stage`.
Первым делом:
```bash
python monitor.py --check-embedder
```
Печатает версии Python, ОС, torch, CUDA, sentence-transformers, transformers, numpy, устройство, имя модели, папку кэша, время загрузки, тестовый encode и итоговый режим. Код выхода 0, если всё ок, 3 если нет.

Защита от падений (segfault в torch/CUDA, например Python 3.14 + WSL2):
- и `--check-embedder`, и обычный запуск сначала грузят модель и делают тестовый encode в отдельном процессе (`python -m a2179.embed_probe`, с `faulthandler`). Если он умер по сигналу (SIGSEGV, SIGABRT), вернул ошибку или висит дольше `embedder.probe_timeout_s`, монитор пишет `[embedder] ... (crashed)` и событие `embedder_probe` с кодом выхода и хвостом stderr в `data/logs/monitor.jsonl`, а дальше работает на regex + TF-IDF. torch в основной процесс при этом не грузится вообще.
- если проверка прошла, модель грузится в основном процессе на том же устройстве, что отчитался дочерний (`device` в выводе и в событии `embedder_ready`). Encode во время прогона идёт в основном процессе: проверка только пропускает или не пропускает эмбеддер. Падение, которое случается только позже на каких то текстах, это не ловит, такого пока не видели.
- выключить проверку: `"probe": false` (на секунды быстрее старт, но без защиты).
- порядок импорта: torch всегда импортируется раньше sentence-transformers (а значит и sklearn с его loky/OpenMP), иначе на части Linux/WSL сборок две копии OpenMP дают segfault.
- Python 3.14 поддерживается, если под него есть сборка torch.
- `"device": "cpu"`: до импорта torch ставится `CUDA_VISIBLE_DEVICES=""` (и в процессе проверки, и в основном, если torch там ещё не импортирован), а `torch.cuda.is_current_stream_capturing` и `transformers.utils.import_utils.is_cuda_stream_capturing` подменяются на «нет» до загрузки модели. Причина: transformers 5.x даже на cpu спрашивает CUDA (is_tracing), и на WSL2 инициализация CUDA падала с segfault. `--check-embedder` печатает строку `cpu guard`.

| stage | что значит | что делать |
|---|---|---|
| `not_installed` | нет sentence-transformers или torch | поставить torch под ОС (ниже), потом `pip install -r requirements-embed.txt` |
| `import_failed` | пакет есть, но не импортируется | NumPy ABI: `pip install "numpy<2"` или новее torch. Windows DLL (WinError 126): Visual C++ Redistributable x64, Python 64 bit |
| `wrong_model` | нет такой модели на HF | проверить `embedder.model` |
| `offline` | модели нет в кэше, а сеть выключена | один запуск с интернетом, без `HF_HUB_OFFLINE=1` и с `local_only: false` |
| `download_failed` | нет сети, прокси, лимит 429 | `HTTPS_PROXY=...`, подождать или задать `HF_TOKEN` |
| `cache_permission` | нет прав на кэш HF | `HF_HOME` на папку с правами записи |
| `load_failed`, `encode_failed` | несовместимые версии | `pip install -U -r requirements-embed.txt`, прислать вывод `--check-embedder` |
| `crashed` | процесс проверки упал по сигналу (обычно torch/CUDA) | драйвер: `nvidia-smi` (максимальная CUDA драйвера), `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"`, `python -c "import torch; torch.zeros(1).cuda()"`. WSL2: драйвер NVIDIA ставится на Windows (внутри WSL не ставить), CUDA сборки torch (cuXXX) не новее, чем у драйвера. Пока можно `"device": "cpu"` |
| `probe_timeout` | проверка висит дольше `probe_timeout_s` | первая загрузка модели долгая: поднять `probe_timeout_s` или запустить ещё раз |
| `probe_failed` | процесс проверки вышел без результата | прислать вывод `--check-embedder` |

Установка torch (сначала torch, потом `requirements-embed.txt`):
```bash
# Linux, WSL2, Windows с NVIDIA: сборка под свою CUDA, команду бери в селекторе на https://pytorch.org/get-started/locally/
pip install torch --index-url https://download.pytorch.org/whl/cu128   # пример для CUDA 12.8
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"   # должно быть True
# по желанию, машина без GPU: CPU сборка, она меньше
pip install "torch>=2.2" --index-url https://download.pytorch.org/whl/cpu
# macOS Apple Silicon (M1 и новее):
pip install "torch>=2.2"
# macOS Intel (x86_64): последний torch для Intel Mac это 2.2.2, он есть только для Python до 3.12 и собран под NumPy 1.x
python3.12 -m venv .venv && source .venv/bin/activate
pip install "torch==2.2.2" "numpy<2" "transformers>=4.41,<4.50" "sentence-transformers>=3.0,<4"
pip install -r requirements.txt "numpy<2"
```
Python 3.13 поддерживается свежими torch (2.5 и новее), 3.14 тоже, если под него есть колесо torch. На Intel Mac бери 3.11 или 3.12.

Устройство: `embedder.device` в `config.json` (или `A2179_DEVICE=cuda`). `auto` берёт `cuda`, если torch её видит, иначе `cpu`. `--check-embedder` показывает, что выбрано и почему.

ONNX: `"backend": "onnx"` (нужны `sentence-transformers>=3.2` и `pip install "optimum[onnxruntime]"`, для GPU `optimum[onnxruntime-gpu]`). Encode тогда идёт через onnxruntime, но sentence-transformers всё равно импортирует torch, так что от падения при импорте torch это не спасает. Следующий шаг, если понадобится: путь совсем без torch (onnxruntime + tokenizers напрямую).

Кэш и офлайн:
- модель (~470 МБ) один раз качается в `~/.cache/huggingface` (Windows: `%USERPROFILE%\.cache\huggingface`). Другая папка: `export HF_HOME=/путь` (Windows: `set HF_HOME=D:\hf`).
- после первой загрузки можно без сети: `export HF_HUB_OFFLINE=1` или `"local_only": true` в `config.json`.
- первый запуск дольше (скачивание плюс загрузка torch, обычно десятки секунд), дальше загрузка модели несколько секунд.
- не хочешь эмбеддер вообще: `"embedder": {"enabled": false}` или `--no-embed`, тогда и сообщений не будет.

Что прислать, если не помогло: вывод `python monitor.py --check-embedder`, полный traceback из `python monitor.py --dry-run --require-embedder -v`, `python --version`, ОС и архитектуру, `pip freeze | grep -iE "torch|sentence|transformers|numpy|huggingface"`.

### Ограничения
- KupujemProdajem часто отвечает 403 ботам и серверным IP («Uskoro ćemo ponovo biti dostupni»). Монитор делает один вежливый повтор и пропускает KP до следующего запуска. С домашнего сербского IP шансы лучше. Разметка KP не проверена вживую, парсер ищет данные в `__NEXT_DATA__` и ссылки вида `/oglas/<id>`. Если KP отдаёт страницу, но 0 объявлений, в логе будет `kp_no_items`, пришли мне кусок HTML.
- Пагинация KP построена на догадке про параметр `page`, разметка KP вживую не проверена.
- Цена «Dogovor» или «Kontakt» сохраняется без цены, придёт только уведомление о новом объявлении. KP в JSON отдаёт такие объявления с ценой 0: это тоже «без цены», бесплатным считается только явное «Besplatno» или «Poklanjam».
- Курс валют задаётся вручную в конфиге.
