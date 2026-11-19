# a2179_monitor

Монитор объявлений: **экран (LCD панель) или крышка в сборе для MacBook Air 13" 2020 Intel A2179** в Сербии.
Источники: KupujemProdajem (поиск), ifix.rs, ihelp.rs, laptop-punjaci.com (страницы товаров и поиск по сайту).
Без LLM. Фильтр гибридный: регулярки + похожесть на эталонные фразы (локальный эмбеддер HF, если установлен, иначе TF-IDF по символьным n-граммам).
Приоритет полноты: лучше лишнее уведомление, чем пропущенное объявление.

## Что ловит и что нет
- Да: A2179, EMC 3302, 661-15389, «MacBook Air 2020 Intel», латиница и кириллица (ekran, displej, lcd, panel, matrica, komplet, kućište, екран, дисплеј, экран, матрица, крышка, в сборе), доноры «za delove / u delove / на запчасти», бесплатные отдачи.
- Нет: только M1 A2337, A2338, чехлы, cover, плёнки и стёкла, клавиатуры, батареи, зарядки, шлейфы, винты, услуги «zamena ekrana», «tražim/kupujem», целые ноутбуки по характеристикам.
- **A1932 (2018/2019):** магазины продают одну и ту же панель как «A1932 | A2179» (например ifix.rs), поэтому объявления только с A1932 по умолчанию тоже ловятся с пометкой «совместимость уточни у продавца». Выключить: `"accept_a1932": false`.

## Уведомления
- 🆕 новое подходящее объявление, 📉 цена упала (по умолчанию на 3% и больше), 🎯 цена ниже цели (по умолчанию **$20**, курс RSD/EUR/USD в конфиге).
- Дубли отсекаются по локальной базе `data/state.sqlite` (история цен и уведомлений там же).
- Канал: Telegram бот (твой токен и chat id через переменные окружения), плюс консоль, `data/alerts.log` и системное уведомление (Linux `notify-send`, macOS).
- Если KupujemProdajem блокирует запросы (403) 6 запусков подряд, придёт одно предупреждение. Магазины при этом проверяются как обычно.

## Установка
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

## Запуск
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

### Параллельная загрузка
Проход почти целиком состоит из ожидания сети, поэтому источники качаются и разбираются параллельно:
- `parallel.workers` (по умолчанию **3**, `--workers`, `A2179_WORKERS`) процессов, в каждом свой цикл asyncio и **одна** `aiohttp.ClientSession` на процесс. Пока одна страница разбирается, остальные запросы уже ждут ответа.
- `parallel.max_concurrency` (по умолчанию **10**, `--max-concurrency`, `A2179_MAX_CONCURRENCY`): сколько GET запросов одновременно в полёте **всего**, по всем процессам вместе, ретраи тоже считаются. Как сделано: общий лимит делится между процессами, у каждого свой `asyncio.Semaphore`, доли в сумме дают ровно лимит (10 на 3 процесса это 4 + 3 + 3). Общей блокировки между процессами нет, так что застрявший процесс не тормозит остальных и больше лимита в полёте не бывает никогда. Цена: процесс, у которого кончилась работа, свои свободные слоты другим не отдаёт. Процессов не бывает больше лимита (каждому минимум один слот) и больше, чем источников.
- `parallel.delay_s` (по умолчанию 0): пауза между стартами запросов к одному сайту внутри процесса. В параллельном режиме вместо паузы `http.min_delay_s` нагрузку держит лимит, если KP начнёт отвечать 429 или 403, уменьши `--max-concurrency` или поставь `delay_s`.
- Результаты собираются в исходном порядке источников, поэтому дедупликация, уведомления, база и метрики те же, что в последовательном режиме. Если KP ответил блокировкой, остальные запросы к KP в этом проходе уже не стартуют (те, что уже в полёте, долетают, но в учёт не идут, как и раньше).
- Нет `aiohttp` (`pip install -r requirements.txt`): монитор пишет об этом одну строку и проходит последовательно.
- В `data/metrics.json` в `notes` видно, как прошла загрузка, например `fetch=parallel processes=3 max_concurrency=10 split=4+3+3`.

### По расписанию
Linux/macOS, `crontab -e`, раз в час:
```
17 * * * * cd /путь/к/a2179_monitor && .venv/bin/python monitor.py >> data/cron.log 2>&1
```
Windows, Планировщик заданий (PowerShell):
```powershell
schtasks /Create /SC HOURLY /TN a2179_monitor /TR "C:\путь\a2179_monitor\.venv\Scripts\python.exe C:\путь\a2179_monitor\monitor.py"
```
Чаще раза в 30 минут не надо: KP и магазины не любят частые запросы.

## Конфиг (`config.json`, все поля необязательны)
| поле | что |
|---|---|
| `target_usd` | цель по цене, $ (по умолчанию 20) |
| `rates.rsd_per_eur`, `rates.usd_per_eur` | курсы для пересчёта |
| `price_drop_min_pct` | минимальное падение цены для 📉 |
| `alert_on` | какие события слать |
| `accept_a1932` | ловить ли объявления только с A1932 |
| `embedder.enabled` | `auto` или `true` (пробовать, при ошибке откат на TF-IDF), `false` (не трогать), `required` (без него выход с кодом 3) |
| `embedder.local_only` | `true`: не ходить в сеть, брать модель только из кэша HF |
| `embedder.model` | модель sentence-transformers, по умолчанию `paraphrase-multilingual-MiniLM-L12-v2` |
| `thresholds` | пороги канала похожести (подобраны eval-скриптом) |
| `sources` | URL товаров, поисков и ключевые слова KP |
| `http` | пауза между запросами к одному сайту, ретраи, таймаут, User-Agent |
| `parallel` | `workers` (3, `0` = последовательно), `max_concurrency` (10, всего по всем процессам), `delay_s` (0), см. «Параллельная загрузка» |

Переменные окружения: `TG_BOT_TOKEN`, `TG_CHAT_ID`, `A2179_TARGET_USD`, `A2179_RSD_PER_EUR`, `A2179_USD_PER_EUR`, `A2179_EMBEDDER`, `A2179_MODEL`, `A2179_WORKERS`, `A2179_MAX_CONCURRENCY`.

## Наблюдаемость
- `data/logs/monitor.jsonl`: структурные логи (каждый запрос, статус, задержка, решение фильтра с причинами).
- `data/metrics.json` и `data/metrics_history.jsonl`: число запросов, ошибки по сайтам и HTTP статусам, задержки (avg/p95/max), сколько объявлений, совпадений, событий и отправок. При `--dry-run` пишется `data/metrics_dryrun.json`.

## Оценка качества
```bash
python eval/run_eval.py              # regex, tfidf, hybrid_tfidf, и embed/hybrid_embed если эмбеддер установлен
python eval/run_eval.py --no-a1932   # если считать A1932 несовместимым
```
Размеченный набор `eval/labeled.jsonl`: 97 заголовков (40 подходящих, 57 нет), сербский латиницей и кириллицей, английский, русский. 38 из них реальные заголовки с KP и сайтов магазинов (19.11.2026), остальные составлены вручную, включая трудные отрицательные (A2337, чехлы, шлейфы LCD, услуги, целые ноутбуки).
Отчёт: `reports/eval_report.md` (precision, recall, F1, матрица ошибок, список ошибок, проверка 2-fold).

Результат без эмбеддера (19.11.2026):

| режим | precision | recall | F1 |
|---|---|---|---|
| regex | 0.975 | 0.975 | 0.975 |
| tfidf (только похожесть) | 0.800 | 1.000 | 0.889 |
| **regex + tfidf (по умолчанию)** | **0.976** | **1.000** | **0.988** |

Честно: регулярки писались, глядя на этот набор, а пороги подобраны на нём же, так что на новых объявлениях цифры будут ниже. 2-fold (подбор на половине, проверка на другой) даёт для гибрида P/R/F1 0.975/0.975/0.975. Эмбеддер с этой машины не проверялся. У себя после установки запусти `python eval/run_eval.py --write-thresholds`: он посчитает метрики для `embed` и `hybrid_embed` и сам впишет подобранные пороги в `config.json` (с пометкой `calibrated`). Пороги TF-IDF и эмбеддера в разных единицах (косинусы у эмбеддера обычно выше), поэтому чужие пороги не подходят. Пока пороги не откалиброваны, монитор пишет об этом при запуске. Пропусков это не добавляет: регулярки работают всегда, эмбеддер только добавляет находки.

## Тесты
```bash
python -m unittest discover -s tests -t .
```
Тесты сеть не трогают, путь эмбеддера проверяется на поддельном модуле sentence_transformers (все ошибки выше): HTML магазинов сохранён в `tests/fixtures` (снимки 19.11.2026), KP заменён синтетической страницей.

## Эмбеддер не работает
Монитор без эмбеддера работает полностью (regex + TF-IDF). Если эмбеддер включён, но не поднялся, при каждом запуске будет три строки `[embedder] ...` с причиной и что делать, а в `data/logs/monitor.jsonl` событие `embedder_unavailable` с полем `stage`.
Первым делом:
```bash
python monitor.py --check-embedder
```
Печатает версии Python, ОС, torch, sentence-transformers, transformers, numpy, имя модели, папку кэша, время загрузки, тестовый encode и итоговый режим. Код выхода 0, если всё ок, 3 если нет.

| stage | что значит | что делать |
|---|---|---|
| `not_installed` | нет sentence-transformers или torch | поставить torch под ОС (ниже), потом `pip install -r requirements-embed.txt` |
| `import_failed` | пакет есть, но не импортируется | NumPy ABI: `pip install "numpy<2"` или новее torch. Windows DLL (WinError 126): Visual C++ Redistributable x64, Python 64 bit |
| `wrong_model` | нет такой модели на HF | проверить `embedder.model` |
| `offline` | модели нет в кэше, а сеть выключена | один запуск с интернетом, без `HF_HUB_OFFLINE=1` и с `local_only: false` |
| `download_failed` | нет сети, прокси, лимит 429 | `HTTPS_PROXY=...`, подождать или задать `HF_TOKEN` |
| `cache_permission` | нет прав на кэш HF | `HF_HOME` на папку с правами записи |
| `load_failed`, `encode_failed` | несовместимые версии | `pip install -U -r requirements-embed.txt`, прислать вывод `--check-embedder` |

Установка torch (сначала torch, потом `requirements-embed.txt`):
```bash
# Linux и Windows, только CPU (иначе pip потянет CUDA на несколько ГБ):
pip install "torch>=2.2" --index-url https://download.pytorch.org/whl/cpu
# macOS Apple Silicon (M1 и новее):
pip install "torch>=2.2"
# macOS Intel (x86_64): последний torch для Intel Mac это 2.2.2, он есть только для Python до 3.12 и собран под NumPy 1.x
python3.12 -m venv .venv && source .venv/bin/activate
pip install "torch==2.2.2" "numpy<2" "transformers>=4.41,<4.50" "sentence-transformers>=3.0,<4"
pip install -r requirements.txt "numpy<2"
```
Python 3.13 поддерживается свежими torch (2.5 и новее). На Intel Mac бери 3.11 или 3.12.

Кэш и офлайн:
- модель (~470 МБ) один раз качается в `~/.cache/huggingface` (Windows: `%USERPROFILE%\.cache\huggingface`). Другая папка: `export HF_HOME=/путь` (Windows: `set HF_HOME=D:\hf`).
- после первой загрузки можно без сети: `export HF_HUB_OFFLINE=1` или `"local_only": true` в `config.json`.
- первый запуск дольше (скачивание плюс загрузка torch, обычно десятки секунд), дальше загрузка модели несколько секунд.
- не хочешь эмбеддер вообще: `"embedder": {"enabled": false}` или `--no-embed`, тогда и сообщений не будет.

Что прислать, если не помогло: вывод `python monitor.py --check-embedder`, полный traceback из `python monitor.py --dry-run --require-embedder -v`, `python --version`, ОС и архитектуру, `pip freeze | grep -iE "torch|sentence|transformers|numpy|huggingface"`.

## Ограничения
- KupujemProdajem часто отвечает 403 ботам и серверным IP («Uskoro ćemo ponovo biti dostupni»). Монитор делает один вежливый повтор и пропускает KP до следующего запуска. С домашнего сербского IP шансы лучше. Разметка KP не проверена вживую, парсер ищет данные в `__NEXT_DATA__` и ссылки вида `/oglas/<id>`. Если KP отдаёт страницу, но 0 объявлений, в логе будет `kp_no_items`, пришли мне кусок HTML.
- Только первая страница поиска KP по каждому ключевому слову.
- Цена «Dogovor» или «Kontakt» сохраняется без цены, придёт только уведомление о новом объявлении.
- Курс валют задаётся вручную в конфиге.
