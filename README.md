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
- `parallel.max_concurrency` (по умолчанию **10**, `--max-concurrency`, `A2179_MAX_CONCURRENCY`): сколько GET запросов одновременно в полёте **всего**, по всем процессам вместе, ретраи тоже считаются. Как сделано: **один общий семафор над процессами** (`multiprocessing.BoundedSemaphore`, отдаётся воркерам при старте, без отдельного процесса Manager). Слот берётся прямо перед запросом и отдаётся, как только ответ прочитан, поэтому процесс без работы слотов не держит и все слоты достаются тем, у кого работа есть. Воркеры берут слот неблокирующим опросом (раз в 5..50 мс), цикл событий не встаёт.
- Ретраи: сетевые ошибки, 5xx и 429 повторяются до `http.retries` раз (по умолчанию **3**) с экспоненциальной паузой и джиттером: `backoff_s * 2^(n-1)`, умноженное на случайное 0.5..1.5, не больше `http.backoff_max_s` (по умолчанию 5, 10, 20 с, потолок 60 с). 404 и прочие 4xx не повторяются, 403 повторяется `http.retry_403` раз, как раньше.
- 429 и лимит сайта: у каждого сайта своя пауза между стартами запросов, общая для всех процессов (AIMD). На каждый 429 пауза умножается на `delay_up` (2.0), но не меньше `delay_on_429_s` (0.5 с), а `Retry-After` в секундах останавливает старты к этому сайту на указанное время. На каждый успешный ответ пауза уменьшается на `delay_down_s` (0.05 с). Границы `delay_min_s` (0) и `delay_max_s` (30 с), стартовая пауза `delay_s` (0). Так скорость сама подходит к границе лимита сайта и держится чуть ниже неё. Выученная пауза переходит в следующий проход того же запуска (`--every`). Хочешь осторожнее: подними `delay_min_s` или `delay_on_429_s`, быстрее отходить после бана: подними `delay_down_s`.
- Результаты собираются в исходном порядке источников, поэтому дедупликация, уведомления, база и метрики те же, что в последовательном режиме. Если KP ответил блокировкой, остальные запросы к KP в этом проходе уже не стартуют (те, что уже в полёте, долетают, но в учёт не идут, как и раньше).
- Нет `aiohttp` (`pip install -r requirements.txt`): монитор пишет об этом одну строку и проходит последовательно.
- В `data/metrics.json` в `notes` видно, как прошла загрузка, например `fetch=parallel processes=3 max_concurrency=10 (one shared limiter)` и выученные паузы `pace kupujemprodajem.com=0.85s`. Счётчики `fetch.retries` и `fetch.rate_limited` считают ретраи и ответы 429, в логе событие `rate_limited` с текущей паузой.

### Пагинация поисков
Поиски магазинов (WooCommerce) и KP читаются **до последней страницы**, пока страницы есть. Каждая страница идёт через тот же общий лимит, паузу на 429 и ретраи.
- Следующая страница: `<link rel="next">`, `<a rel="next">` или WooCommerce `a.next.page-numbers`. Для KP ещё ссылка на тот же путь с `page=N+1`, а если её нет, адрес с `page=N+1` (разметка KP вживую не проверена, параметр задаёт `pagination.kp_page_param`, пустая строка выключает догадку).
- Стоп: пустая страница (`empty`), нет ссылки на следующую (`no_next`), на странице нет новых объявлений (`no_new`), 404 или 410 после первой страницы (`end`, так WooCommerce отвечает за последней), другая ошибка (`error`), KP заблокировал (`blocked`, дальше по KP в этом проходе не идём, как и раньше), ссылка ведёт на уже прочитанную страницу (`loop`) или на другой сайт (`off_host`).
- `pagination.max_pages` (по умолчанию **500**) только аварийный потолок на один поиск (`ceiling`), низкого лимита страниц нет. `pagination.enabled: false` вернёт чтение только первой страницы.
- В метриках `pagination.shop_search.pages`, `pagination.kp.pages` и `pagination.stop.<причина>`.

### Отслеживаемые карточки
Если в результатах поиска магазина нашлась подходящая карточка (те же регулярки и похожесть по смыслу, что для уведомлений), она попадает в постоянный список `data/tracked.shelve` и дальше каждый проход проверяется как `shop_products`: цена, наличие, статус. Так товар не теряется, даже если пропадёт из поиска. Объявления KP так не отслеживаются (парсера страницы объявления KP нет, их покрывает поиск KP со всеми страницами).
- Запросы к карточкам идут через тот же общий лимит и паузу на 429.
- Хранилище `shelve` (файл выбирает бэкенд dbm, например `tracked.shelve` или `tracked.shelve.db`). Защита от гонок: **один** `multiprocessing.Lock`, его, как и семафор запросов, главный процесс создаёт и отдаёт воркерам при старте. Любое чтение или запись берёт этот замок, открывает shelve, делает своё и закрывает его до того, как замок отпущен, так что shelve никогда не открыт дважды и запись уже на диске, когда к нему подходит следующий процесс. Воркеры пишут только результат проверки своих карточек (запись идёт в отдельном потоке, цикл событий не встаёт), новые карточки добавляет главный процесс после фильтра. `--dry-run` ничего не пишет.
- Замок защищает процессы одного запуска. Два отдельных запуска на одной папке `data` одновременно не защищены: не запускай cron, пока крутится `--every`.
- Карточка, которая `tracked.drop_after_misses` проходов подряд (по умолчанию 3) отвечает 404 или 410, помечается `gone` и больше не запрашивается. Из файла она не удаляется.
- `tracked.enabled: false` выключает список целиком, `tracked.file` меняет имя файла в `data_dir`.

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

## Наблюдаемость
- `data/logs/monitor.jsonl`: структурные логи (каждый запрос, статус, задержка, решение фильтра с причинами).
- `data/tracked.shelve`: отслеживаемые карточки (см. выше).
- `data/metrics.json` и `data/metrics_history.jsonl`: число запросов, ошибки по сайтам и HTTP статусам, задержки (avg/p95/max), сколько объявлений, совпадений, событий и отправок. При `--dry-run` пишется `data/metrics_dryrun.json`.

## Оценка качества
```bash
python eval/run_eval.py              # regex, tfidf, hybrid_tfidf, и embed/hybrid_embed если эмбеддер установлен
python eval/run_eval.py --no-a1932   # если считать A1932 несовместимым
python eval/sweep.py                 # перебор порогов и fusion: BM25, TF-IDF, MiniLM, RRF, явные признаки (отчёт reports/sweep_2026-11-19.md)
python eval/sweep.py --compute-dense # то же, плюс посчитать косинусы MiniLM (нужен эмбеддер)
```
Размеченный набор `eval/labeled.jsonl`: 97 заголовков (40 подходящих, 57 нет), сербский латиницей и кириллицей, английский, русский. 38 из них реальные заголовки с KP и сайтов магазинов (19.11.2026), остальные составлены вручную, включая трудные отрицательные (A2337, чехлы, шлейфы LCD, услуги, целые ноутбуки).
Отчёт: `reports/eval_report.md` (precision, recall, F1, матрица ошибок, список ошибок, проверка 2-fold).

Результат без эмбеддера (19.11.2026):

| режим | precision | recall | F1 |
|---|---|---|---|
| regex | 0.975 | 0.975 | 0.975 |
| tfidf (только похожесть) | 0.800 | 1.000 | 0.889 |
| **regex + tfidf (по умолчанию)** | **0.976** | **1.000** | **0.988** |

Честно: регулярки писались, глядя на этот набор, а пороги подобраны на нём же, так что на новых объявлениях цифры будут ниже. 2-fold (подбор на половине, проверка на другой) даёт для гибрида P/R/F1 0.975/0.975/0.975. Эмбеддер с этой машины не проверялся. У себя после установки запусти `python eval/run_eval.py --write-thresholds`: он посчитает метрики для `embed` и `hybrid_embed` и сам впишет подобранные пороги в `config.json` (с пометкой `calibrated`). Пороги TF-IDF и эмбеддера в разных единицах (косинусы у эмбеддера обычно выше), поэтому чужие пороги не подходят. Пока пороги не откалиброваны, монитор пишет об этом при запуске. Пропусков это не добавляет: регулярки работают всегда, эмбеддер только добавляет находки. Итог перебора 19.11 (`reports/sweep_2026-11-19.md`): лучший прозрачный вариант hybrid_tfidf с pos 0.34, margin 0.16, эмбеддер на этом наборе даёт 8 FP против 1, поэтому в `config.example.json` он выключен.

## Тесты
```bash
python -m unittest discover -s tests -t .
```
Тесты сеть не трогают, путь эмбеддера проверяется на поддельном модуле sentence_transformers (все ошибки выше): HTML магазинов сохранён в `tests/fixtures` (снимки 19.11.2026), KP заменён синтетической страницей.
Реальные данные: `tests/fixtures/dryrun_2026-11-19.jsonl` (17 совпадений из живого `--dry-run --list` с сербского IP: заголовок, цена, $, источник, ссылка) и `tests/fixtures/kp_next_data_dryrun.html` (12 объявлений KP из того же прогона), тесты в `tests/test_real_dryrun.py`: каждое реальное объявление должно ловиться regex и гибридом, KP разбирается с теми же ценами и городами, «0 RSD» не даёт 🎯.

## Эмбеддер не работает
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

## Ограничения
- KupujemProdajem часто отвечает 403 ботам и серверным IP («Uskoro ćemo ponovo biti dostupni»). Монитор делает один вежливый повтор и пропускает KP до следующего запуска. С домашнего сербского IP шансы лучше. Разметка KP не проверена вживую, парсер ищет данные в `__NEXT_DATA__` и ссылки вида `/oglas/<id>`. Если KP отдаёт страницу, но 0 объявлений, в логе будет `kp_no_items`, пришли мне кусок HTML.
- Пагинация KP построена на догадке про параметр `page`, разметка KP вживую не проверена.
- Цена «Dogovor» или «Kontakt» сохраняется без цены, придёт только уведомление о новом объявлении. KP в JSON отдаёт такие объявления с ценой 0: это тоже «без цены», бесплатным считается только явное «Besplatno» или «Poklanjam».
- Курс валют задаётся вручную в конфиге.
