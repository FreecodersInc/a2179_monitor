# CI на своём раннере (эксперименты на CUDA)

Workflow: [`ci/experiments.yml`](../ci/experiments.yml), тонкая обёртка. Вся логика в `ci/run.sh` и скриптах рядом. Решения: [ADR-0008](adr/0008-ci-on-self-hosted-runner.md) (свой раннер), [ADR-0009](adr/0009-thin-workflow-wrapper.md) (тонкая обёртка без actions), [ADR-0011](adr/0011-ci-secrets.md) (секреты).

## Включить или обновить (на твоей стороне)
Токен Telega не может писать в `.github/workflows`, поэтому исходник лежит в `ci/experiments.yml`, а активная копия в `.github/workflows/experiments.yml`. Копируешь ты, из корня репы:
```
cp ci/experiments.yml .github/workflows/experiments.yml && git commit -am 'ci: experiments workflow with environment experiment' && git push
```
YAML только забирает пушнутый коммит и запускает `bash ci/run.sh`, всё остальное Telega меняет в `ci/` обычным пушем. H-11 / EXP-029 это шаг `exp029` в `ci/run_experiments.py` (заявка `ci/requests/runner_experiments.json`), копия workflow для него не нужна. Копия нужна, только когда меняется то, что живёт в самом YAML: триггеры, права, `environment`, имена секретов. Последняя такая правка: `environment: experiment` и секреты `TG_BOT_TOKEN`, `TG_CHAT_ID`, `A2179_CI_SECRETS`. Новые секреты дальше без копии: кладёшь их JSON объектом в один секрет `A2179_CI_SECRETS` (см. «Секреты»). Новые настройки приходят переменными репы `A2179_CI_*` (весь `vars` уходит в `ci/run.sh` как JSON), YAML для них не трогается. Если `.github/workflows/experiments.yml` вдруг разойдётся с `ci/experiments.yml`, это видно в каждом результате (`workflow_in_sync: false`), тогда и только тогда нужна новая копия.

## Почему первый прогон был красный
Run #1 (коммит 4ee8426) упал на старте (`startup_failure`, ни одного job): политика организации FreecodersInc разрешает только actions из репозиториев FreecodersInc, а старый YAML звал `actions/checkout@v4` и `actions/upload-artifact@v4`. Теперь в YAML нет ни одного `uses:`: checkout делается обычным `git fetch` пушнутого sha, артефакта нет, отчёты коммитятся рядом с результатом.

## Что делает прогон
На каждый push в `main` этой репы (и только на него):
1. YAML: `git fetch` пушнутого sha из этой репы в рабочую папку раннера, `checkout -B main`, `git clean`, `exec bash ci/run.sh`. Токен только в окружении процесса (`GIT_CONFIG_*`), в `.git/config` не пишется.
2. `ci/run.sh`:
   - переменные репы `A2179_CI_*` и `HF_*` из JSON в окружение (`ci/vars_env.py`, значения экранируются)
   - Hugging Face офлайн, если нет `A2179_CI_ALLOW_DOWNLOAD=true`
   - `ci/setup_venv.sh`: постоянный venv вне чекаута (`~/.cache/a2179-ci/venv`), `pip install -r requirements.txt` только когда файл поменялся, `requirements-embed.txt` только если в venv уже стоит torch. Лог в `ci_out/setup.log`
   - `ci/run_experiments.py` в этом venv, каждый шаг в своём процессе, с кодом выхода, временем и хвостом вывода:
     - `ci/env_probe.py`: python, torch, CUDA, имя GPU
     - unit тесты
     - `monitor.py --check-embedder` (модель сначала грузится в дочернем процессе, ADR-0005)
     - `eval/run_eval.py` (с эмбеддером, если проверка прошла, иначе `--no-embed`)
     - `eval/sweep.py --compute-dense` (иначе без dense)
     - если dense шаг всё равно упал, он перезапускается без эмбеддера и помечается `fallback`
     - `eval/plots.py` (шаг `plots`, `advisory`: его падение видно в результате, но крестик не краснеет): графики EDA и метрики ранжирования (P@R1, P@R95, R@k, nDCG) по оценкам свипа
     - `eval/review_run.py` (шаг `review`, тоже `advisory`): оба кейса в `reports/<дата>/<кейс>/` и в `results/<base>/reports/`. Экран из `eval/candidates_*.jsonl`, вакансии из официальных API Remotive и Remote OK (кэш 24 часа в `~/.cache/a2179-ci/jobs`, не чаще 4 запросов Remotive в сутки). В `matches.csv` / `all.csv` есть title, company, location, score, decision, url, fetched_at
   - `ci/write_results.py` системным python (только stdlib, работает даже если venv сломался): `results/<UTC дата>_<short sha>.json` и папка `results/<UTC дата>_<short sha>/` с отчётами (eval_report.md/json, sweep.tables.md, dense_scores.jsonl, setup.log, reviews в `reports/<дата>/<кейс>/`) и графиками (`01_dataset.png` … `09_errors_by_kind.png`, `plots.md`, `plots.json`). Графики вклеены в конец `sweep.tables.md`, так что таблицы и картинки читаются одним отчётом. Коммит только этих файлов с `[skip ci]`, push в `main`, при отказе `git pull --rebase` и ещё раз, до 5 попыток
3. Красный крестик: если упала установка venv, unit тесты, интеграционный тест Telegram или пуш результатов. Результат пишется в любом случае.

Петли нет: `results/**` в `paths-ignore`, в сообщении `[skip ci]`, и push через `GITHUB_TOKEN` вообще не запускает workflow.

Локально (без пуша): `A2179_CI_NO_PUSH=1 A2179_CI_RESULTS_DIR=/tmp/res bash ci/run.sh`.

## Почему прогон 3b678fa был красный
Опять тест, не код: `test_ci ... TestOwnVenvWithTorch.test_install_flag_is_noop_when_torch_imports`. В 3b678fa кэш модели стал включён по умолчанию, а этот тест подкладывает фейковый torch в venv, сделанный из python раннера. В venv Лёни настоящий `huggingface_hub`, поэтому шаг кэша пошёл в сеть за фейковой моделью `m`, получил 401 и уронил setup внутри теста. Починка: тест выключает шаг кэша (`A2179_CI_CACHE_MODEL=0`), а сам `setup_venv.sh` больше не падает, если скачать модель не вышло (warning, dense дальше откатывается как без модели). Новый тест проверяет это с подложным `huggingface_hub`, который падает без сети. Плюс тесты CI и графиков читают замороженные отчёты из `tests/fixtures/`, а не `reports/`, которые Лёня перезаписывает своими прогонами (иначе следующий прогон был бы красным на `ev["dataset"]["n"] == 99`).

## Почему второй прогон был красный (ebaed9c)
Сам CI отработал: venv, eval, свип, результат закоммичен (`results/2026-11-19_ebaed9c.json`). Красным был шаг unit тестов, 3 теста из 179, все из за моих тестов, не из за кода:
- `test_ci ... test_writes_result_and_reports`: тест делал «venv» из симлинка на `sys.executable`. На раннере это python из venv, а симлинк вне своего venv запускает голый системный python без пакетов, вложенный eval падал на импорте. Теперь там обёртка `exec <python> "$@"`.
- `test_parallel ... test_idle_processes_hold_no_slots` (4.3 s при пороге 1.3) и `test_local_server` (6.0 s при 2.4): мерили общее время вместе со стартом процессов, а spawn на WSL2 с Python 3.14 медленный. Теперь сначала тот же проход без задержек (база), сравнивается только разница.

## Почему прогон 42447e5 был красный
Код и эксперименты отработали (eval, свип, графики, результат закоммичен), красным был один тест: `test_ci ... test_writes_result_and_reports` гонял весь `ci/run.sh` с таймаутом 90 s. На MX130 этот вложенный прогон импортировал настоящий torch и рисовал настоящие графики, и не влез. Починка: тест прячет от вложенных шагов torch, sentence-transformers и matplotlib (sitecustomize во временной папке), проверяет только проводку `run.sh`, таймаут 300 s как запас. Настоящие шаги и так идут в самом CI.

Путь с папкой `~` (`actions-runner/~/a2179_monitor/...`) это рабочая папка самого раннера: при `./config.sh` в `--work` попала строка `~/a2179_monitor` в кавычках, и раннер сделал папку с именем `~` внутри `actions-runner/`. Переменная `A2179_CI_VENV` тут ни при чём, она абсолютная. Работе это не мешает. Скрипты CI теперь сами раскрывают ведущий `~/` в `A2179_CI_VENV` и `A2179_CI_PYTHON` (через `$HOME`, без eval) и пишут warning, если рабочая папка содержит `/~/`. Убрать совсем: остановить раннер и в `actions-runner/.runner` поменять `"workFolder"` на `"_work"` (или абсолютный путь), потом запустить снова.

## Эмбеддеры в CI на CPU (после зависания 2026-11-19)
Несколько эмбеддеров подряд на MX130 (2 ГБ) вылезли из видеопамяти в общую RAM через PCI и повесили всю машину. Теперь по умолчанию:
- `A2179_CI_DEVICE=cpu`: CUDA спрятана от шагов (`CUDA_VISIBLE_DEVICES=`), GPU никто не трогает. `cuda` вернёт GPU
- `A2179_CI_THREADS` (по умолчанию половина ядер, не больше 4) для torch, MKL, OpenBLAS
- `A2179_CI_EMBED_BATCH=16`
- модели строго по одной, между ними `del` и `gc`, в логе строка `dense i/N <модель>` на каждую
- оценки моделей кэшируются в `~/.cache/a2179-ci/dense` (`A2179_CI_DENSE_CACHE`), ключ: id модели, префиксы, прототипы и строки (id + текст). Повторный прогон на тех же данных модель не грузит, в логе `reused cached scores`
- `A2179_CI_EMBEDDERS=cached-small`: только модели из кэша до `A2179_MAX_PARAMS_M=300` M параметров (minilm, mpnet, e5-small, e5-base и варианты, bge-base-en, e5-base-en). bge-m3, e5-large, e5-large-instruct, LaBSE только если назвать их явно, например `A2179_CI_EMBEDDERS=cached-small,bge-m3,e5-large,e5-large-instruct,labse` (группы и имена можно смешивать)

## torch и модель (опционально, из CI)
Ставить torch из CI тяжело (несколько ГБ), поэтому только по явной переменной репы. Settings → Secrets and variables → Actions → Variables → New repository variable:
```
A2179_CI_INSTALL_TORCH = cu126      # или cu128, cpu, rocm6.2 (сборка под твою CUDA)
```
Следующий прогон (шаг `setup_venv`, лог в `results/<base>/setup.log`):
1. если torch в venv нет, ставит `torch` с `https://download.pytorch.org/whl/<значение>`, один раз, уже стоящий torch не трогает
2. ставит `requirements-embed.txt`
3. кладёт модель (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, или `A2179_CI_MODEL`) в кэш Hugging Face: `HF_HOME`, если задан переменной, иначе обычный `~/.cache/huggingface` пользователя раннера (тот же кэш, что у твоих ручных запусков). Через `huggingface_hub.snapshot_download`, torch при этом не импортируется. Если модель уже в кэше, ничего не качает.

Скачивание разрешено только этому шагу, все остальные шаги идут с `HF_HUB_OFFLINE=1`. Шаг кэша модели включён по умолчанию (`A2179_CI_CACHE_MODEL=1`): если модель уже в кэше, он ничего не качает. Выключить: `A2179_CI_CACHE_MODEL=0`.

Свой готовый venv (так сейчас: `A2179_CI_VENV=/home/vessyl-runner/a2179_monitor/.venv`, torch + CUDA, Python 3.14): копировать его не надо, venv не переносится (в `bin/` и `pyvenv.cfg` зашиты абсолютные пути), достаточно переменной с путём. Что CI делает с этим venv:
- не создаёт заново, если там есть `bin/python` (`A2179_CI_PYTHON` нужен только для нового venv)
- **ставит в него `requirements.txt` и `requirements-embed.txt` через pip, когда они поменялись** (отметка `.stamp_<файл>` внутри venv, первый прогон ставит всегда). У пользователя раннера должны быть права на запись
- torch не трогает: если torch импортируется, `A2179_CI_INSTALL_TORCH` ничего не делает, так что переменную можно не ставить
- модель в кэш по умолчанию (`A2179_CI_CACHE_MODEL=0` выключает; если уже лежит в `~/.cache/huggingface`, ничего не качается)

## Секреты (GitHub environment `experiment`)
Job идёт с `environment: experiment`, оттуда приходят `TG_BOT_TOKEN` и `TG_CHAT_ID`. Если у environment стоят правила защиты (обязательный ревьюер, ограничение веток), прогон ждёт их, как и любой deploy. Ветка `main` должна быть разрешена.
- Секреты передаются **по имени**, не `toJSON(secrets)`: тот отдал бы шагу все секреты репы и организации разом, и любая новая утечка в логах стала бы утечкой всего.
- Чтобы новые секреты не требовали копии YAML: один секрет `A2179_CI_SECRETS` с JSON объектом, например `{"SECRET_DB_URL": "...", "TG_SECOND_CHAT": "..."}`. Берутся только имена `TG_*`, `SECRET_*`, `A2179_SECRET_*` (так JSON не может переписать `PATH`, `GIT_*`, `GITHUB_*` или настройки `A2179_CI_*`). Каждое значение маскируется в логе (`::add-mask::`), GitHub сам маскирует только секрет целиком.
- Значения никогда не печатаются и не пишутся: в результате только `secrets: {"TG_BOT_TOKEN": "present", ...}`. Все хвосты вывода, `setup.log`, отчёты и сам JSON результата проходят через `ci/redact.py`, значения длиной от 4 символов заменяются на `***`.
- Секреты получает только шаг `tg_integration`. Unit тесты, eval и свип идут без них, так что никакой тест не отправит сообщение случайно.
- `tg_integration` (`integration/test_telegram.py`): без обоих секретов пропускается (в результате `skipped`). С ними только чтение: `getMe` (токен живой) и `getChat` (бот видит чат). Плюс одно сообщение через Alerter, по умолчанию включено (`A2179_CI_TG_SEND=0` выключает). Красный крестик, если проверка не прошла.

## Настройка раннера (один раз)
1. Отдельный пользователь без sudo, например `a2179ci`. Раннер, venv, кэш HF живут в его `$HOME`.
2. Раннер регистрировать **на уровне репы** (Settings → Actions → Runners → New self-hosted runner), не на организацию, чтобы другие репы его не видели. При `./config.sh` добавить метку:
   ```
   ./config.sh --url https://github.com/FreecodersInc/a2179_monitor --token <token со страницы> --name leonid-wsl --labels a2179-gpu
   ```
   Workflow ждёт `runs-on: [self-hosted, linux, a2179-gpu]`, первые две метки раннер ставит сам. Метку можно добавить и потом на странице раннера.
3. Запуск от этого пользователя: `./run.sh` (в tmux) или как сервис. `sudo ./svc.sh install a2179ci` делаешь ты один раз как админ, сам раннер работает под `a2179ci` без sudo. На WSL2 без systemd проще `./run.sh`.
4. torch и модель: проще всего переменная `A2179_CI_INSTALL_TORCH=cu126` (раздел «torch и модель»). Руками, от пользователя раннера:
   ```
   ~/.cache/a2179-ci/venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu126   # твоя CUDA
   ```
   Следующий прогон сам доставит `requirements-embed.txt`, модель тоже (кэш модели включён по умолчанию).
5. Прогон по умолчанию офлайн (`HF_HUB_OFFLINE=1`), качать может только шаг кэша модели.
6. Политика actions организации трогать не надо: workflow не использует ни одного action.
7. Settings → Actions → General → Workflow permissions: если пуш результатов падает с 403, включить «Read and write permissions». Если на `main` стоит защита ветки с обязательными PR, пуш результатов отклонится, тогда разрешить GitHub Actions обходить правило или поставить переменную `A2179_CI_NO_PUSH=1` (результаты останутся только в `ci_out/` на раннере).

## Переменные репы (Settings → Secrets and variables → Actions → Variables, всё необязательно)
| переменная | по умолчанию | что |
|---|---|---|
| `A2179_CI_DENSE` | `auto` | `auto` dense если есть, `on` то же, но причина отказа в отчёте, `off` без эмбеддера |
| `A2179_CI_VENV` | `~/.cache/a2179-ci/venv` | где venv (свой готовый тоже можно, CI доставит в него requirements) |
| `A2179_CI_PYTHON` | `python3` | чем создать venv, например `python3.14` |
| `A2179_CI_ALLOW_DOWNLOAD` | не задано | `true` разрешает скачивать модель с Hugging Face |
| `A2179_CI_STEP_TIMEOUT` | `1800` | секунд на один шаг, весь job 180 минут |
| `HF_HOME` | `~/.cache/huggingface` | где кэш моделей |
| `A2179_CI_INSTALL_TORCH` | не задано | `cu126` / `cu128` / `cpu` / `rocm6.2`: поставить torch в venv, если его нет, и закэшировать модель |
| `A2179_CI_CACHE_MODEL` | `1` | положить модель в `HF_HOME`, если её там нет; `0` выключает |
| `A2179_CI_MODEL` | модель из конфига | какую модель кэшировать |
| `A2179_CI_MODELS` | не задано | список в кэш через запятую: id, короткие имена или группы из `eval/embedders.json` (`multilingual,english` или `e5-base,bge-m3`), `all` все. Качаются только нужные файлы, уже лежащие в кэше пропускаются, упавшая модель не роняет остальные и setup. Вручную: `python scripts/download_embedders.py multilingual english` |
| `A2179_CI_EMBEDDERS` | `cached` | какие модели свип сравнивает рядом с MiniLM (EXP-003): `cached` все из реестра, что уже в кэше, `none`, `all` или список. Ничего не качает |
| `A2179_CI_TG_SEND` | `1` | интеграционный тест шлёт одно сообщение через Alerter; `0` только читает |
| `A2179_CI_PLOTS` | `1` | графики и метрики ранжирования после свипа (около 1 МБ PNG на прогон в `results/`); `0` выключает |
| `A2179_CI_REVIEW` | `1` | шаг review: отчёты ручной проверки по обоим кейсам; `0` выключает |
| `A2179_CI_FETCH_JOBS` | `auto` | `auto` один запрос к Remotive/Remote OK, если кэш старше 24 часов, `never` только кэш, `always` с дневным лимитом Remotive |
| `A2179_CI_TARGETS` | `a2179_screen,de_jobs` | какие кейсы пишет review |
| `A2179_JOBS_CACHE` | `~/.cache/a2179-ci/jobs` | где лежит кэш ответов API вакансий |
| `A2179_CI_NO_PUSH` | не задано | `1`: результат не коммитится, остаётся в `ci_out/` |

Любая новая `A2179_CI_*` переменная доходит до `ci/run.sh` без правки YAML.

## Безопасность
- Триггер только `push` в `main` этой репы. Нет `pull_request`, `pull_request_target`, `workflow_run`, `schedule`, `workflow_dispatch`: код из PR и форков на раннере не исполняется. Плюс `if: github.repository == 'FreecodersInc/a2179_monitor'`.
- Код на твоей машине может запустить только тот, у кого есть push в `main`. Это и есть граница доверия.
- `permissions: contents: write`, больше ничего. Секреты: `GITHUB_TOKEN`, `TG_BOT_TOKEN`, `TG_CHAT_ID`, `A2179_CI_SECRETS`, все только в окружении шага, значения вычищаются из всего, что пишется (раздел «Секреты»).
- Ни одного стороннего action: исполняется только код этой репы на пушнутом sha.
- Сеть: workflow ходит в GitHub (fetch и push). pip ходит в PyPI, только когда поменялись requirements, и на download.pytorch.org только с `A2179_CI_INSTALL_TORCH`. Hugging Face только в шаге кэша модели или при `A2179_CI_ALLOW_DOWNLOAD=true`. api.telegram.org только шаг `tg_integration` с секретами. Одно сообщение в этот чат по умолчанию (`A2179_CI_TG_SEND=0` выключает). Шаг review при пустом или старом кэше один раз в сутки ходит на `remotive.com/api` и `remoteok.com/api` (официальные JSON API, кэш в `$HOME`). LinkedIn, Indeed и Glassdoor не трогаем.
- Без sudo, пишет только в чекаут и `$HOME` пользователя раннера.
- Один прогон за раз (`concurrency`), следующий ждёт в очереди, не отменяет текущий.

## Где результаты
`results/<UTC дата>_<short sha>.json` и рядом папка с отчётами, формат в [`results/README.md`](../results/README.md). Файлы ручной проверки ещё пишутся в `reports/<дата>/a2179_screen/` и `reports/<дата>/de_jobs/` (matches.csv / all.csv со ссылками) и коммитятся тем же `[skip ci]` коммитом.

## Почему 634fa55 красный (тесты и кэш оценок)
`tests` упал на `TestOomFallback`: CI экспортирует `A2179_DENSE_CACHE` всем шагам, тест в прогоне 247d60c записал свои фейковые оценки в кэш раннера, и в следующем прогоне `compute_dense` взял их из кэша и не вызвал эмбеддер (devices `[]`). Теперь `tests/__init__.py` убирает `A2179_DENSE_CACHE` из окружения тестов, тесты кэша ставят временную папку сами. Мусорные записи в `~/.cache/a2179-ci/dense` безвредны (ключ по строкам `a`, `b`, с реальным набором не совпадает). Сам кэш работает: sweep 90 s вместо 346 s.

## Какие шаги гонит push (ADR-0026)
`ci/run.sh` первым делом зовёт `ci/select_steps.py`. База сравнения: последний коммит, у которого уже есть `results/<дата>_<sha>.json` (то, что CI реально проверил), иначе `HEAD~1`.
* **skip**: менялись только `docs/`, `experiments/`, `work/`, `reports/`, `results/`, `*.md`. `run.sh` выходит с 0 до venv, коммита результатов нет. Коммиты только с документами всё равно помечаем `[skip ci]`, это экономит запуск раннера.
* **full**: тронут общий код (`a2179/`, `ci/`, `monitor.py`, `targets/`, `eval/run_eval.py`, `eval/sweep.py`, `eval/plots.py`, `eval/embedders.json`, `eval/labeled.jsonl`, `tests/helpers.py`, `tests/__init__.py`, `tests/fixtures/`). Полная регрессия и все шаги экспериментов.
* **targeted**: всё остальное. Lint (`compileall`), только тесты тронутых файлов (тест с тем же именем или импортирующий модуль) и только эксперимент коммита (например `eval/de_jobs/` даёт шаги `eval[de_jobs]`, `sweep[de_jobs]`, `review`).
* `A2179_CI_MODE=full` в переменных репо форсирует полный прогон.
* выбор пишется в `steps.json` (`select`) и в лог.

Локально перед каждым push: lint и быстрые тесты тронутых файлов. Красный CI, который не чинится за 15 минут, откатываем revert коммитом.

### Что поправить в workflow руками (наш токен .github/workflows менять не может)
Вставить в `.github/workflows/experiments.yml` (и в копию `ci/experiments.yml`) вместо текущих `on:` и `concurrency:`:

```yaml
on:
  push:
    branches: [main]
    paths-ignore:
      - "results/**"
      - "docs/**"
      - "reports/**"
      - "experiments/**"
      - "work/**"
      - "**/*.md"

concurrency:
  group: experiments-${{ github.ref }}
  cancel-in-progress: true
```
Тогда коммит только с документами раннер вообще не будит, а новый push отменяет идущий прогон старого.

## Свежие объявления через CI (ADR-0022)
Чтобы раннер собрал свежую пачку A2179, закоммить `ci/requests/fetch_listings.json` с новым `id` (`max_pages`, `time_budget_s` по желанию). Шаг `fetch_listings` отработает один раз и положит строки в `results/<base>/reports/fresh_a2179/<id>/candidates.jsonl` и `.csv`. Выключить: переменная `A2179_CI_FETCH_LISTINGS=off`.

Разовые тяжёлые эксперименты на раннере (кэш моделей, CUDA) заказываются так же: `ci/requests/runner_experiments.json` с `id` и списком `steps` (`exp028`, `exp029` и т.д.). Каждый шаг один раз на `id`: когда в `results/*/reports/runner_exp/<id>/` уже есть его файл, шаг пропускается. Выключить: `A2179_CI_RUNNER_EXP=off`.

## База компаний через CI (T-038)

`companies.csv` собирается на раннере, не локально. Заявка `ci/requests/company_base.json` с новым `id` один раз запускает шаг `company_base` (`eval/t038_company_base.py`): списки remoteintech/remote-jobs и каталог YC, проба слагов Greenhouse, Lever, Ashby и Workable, вакансии с живых бордов, геофильтр (удалёнка из Сербии) и фильтр по CV, ранжирование ансамблем EXP-030. Результат: `results/<base>/reports/company_base/<id>/companies.csv`, `vacancies_ranked.csv`, `summary.json`. Алертов в Telegram нет. Выключить: `A2179_CI_COMPANY_BASE=off`.

Ежедневная проверка живых бордов: `ci/requests/board_liveness_daily.json`, отчёт в `results/*/reports/company_base/liveness/<дата>/`. Три отказа подряд — борд уходит в спячку. Если в тот же день уже отработал `company_base`, дневная проверка не дублируется.
