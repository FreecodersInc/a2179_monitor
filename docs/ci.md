# CI на своём раннере (эксперименты на CUDA)

Workflow: [`ci/experiments.yml`](../ci/experiments.yml), тонкая обёртка. Вся логика в `ci/run.sh` и скриптах рядом. Решения: [ADR-0008](adr/0008-ci-on-self-hosted-runner.md) (свой раннер), [ADR-0009](adr/0009-thin-workflow-wrapper.md) (тонкая обёртка без actions).

## Включить или обновить (на твоей стороне)
Токен Telega не может писать в `.github/workflows`, поэтому исходник лежит в `ci/experiments.yml`, а активная копия в `.github/workflows/experiments.yml`. Копируешь ты, из корня репы:
```
cp ci/experiments.yml .github/workflows/experiments.yml && git commit -am 'ci: thin experiments workflow' && git push
```
После тонкой обёртки копировать больше не нужно: YAML только забирает пушнутый коммит и запускает `bash ci/run.sh`, всё остальное Telega меняет в `ci/` обычным пушем. Новые настройки приходят переменными репы `A2179_CI_*` (весь `vars` уходит в `ci/run.sh` как JSON), YAML для них не трогается. Если `.github/workflows/experiments.yml` вдруг разойдётся с `ci/experiments.yml`, это видно в каждом результате (`workflow_in_sync: false`), тогда и только тогда нужна новая копия.

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
   - `ci/write_results.py` системным python (только stdlib, работает даже если venv сломался): `results/<UTC дата>_<short sha>.json` и папка `results/<UTC дата>_<short sha>/` с отчётами (eval_report.md/json, sweep.tables.md, dense_scores.jsonl, setup.log). Коммит только этих файлов с `[skip ci]`, push в `main`, при отказе `git pull --rebase` и ещё раз, до 5 попыток
3. Красный крестик: если упала установка venv, unit тесты или пуш результатов. Результат пишется в любом случае.

Петли нет: `results/**` в `paths-ignore`, в сообщении `[skip ci]`, и push через `GITHUB_TOKEN` вообще не запускает workflow.

Локально (без пуша): `A2179_CI_NO_PUSH=1 A2179_CI_RESULTS_DIR=/tmp/res bash ci/run.sh`.

## Настройка раннера (один раз)
1. Отдельный пользователь без sudo, например `a2179ci`. Раннер, venv, кэш HF живут в его `$HOME`.
2. Раннер регистрировать **на уровне репы** (Settings → Actions → Runners → New self-hosted runner), не на организацию, чтобы другие репы его не видели. При `./config.sh` добавить метку:
   ```
   ./config.sh --url https://github.com/FreecodersInc/a2179_monitor --token <token со страницы> --name leonid-wsl --labels a2179-gpu
   ```
   Workflow ждёт `runs-on: [self-hosted, linux, a2179-gpu]`, первые две метки раннер ставит сам. Метку можно добавить и потом на странице раннера.
3. Запуск от этого пользователя: `./run.sh` (в tmux) или как сервис. `sudo ./svc.sh install a2179ci` делаешь ты один раз как админ, сам раннер работает под `a2179ci` без sudo. На WSL2 без systemd проще `./run.sh`.
4. torch под твою CUDA в venv раннера (один раз, от `a2179ci`):
   ```
   python3 -m venv ~/.cache/a2179-ci/venv   # или дождись первого прогона, он создаст
   ~/.cache/a2179-ci/venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128   # твоя CUDA
   ```
   Следующий прогон сам доставит `requirements-embed.txt`.
5. Модель в кэш HF (один раз), потому что прогон по умолчанию офлайн (`HF_HUB_OFFLINE=1`):
   ```
   ~/.cache/a2179-ci/venv/bin/python -c "from sentence_transformers import SentenceTransformer as S; S('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"
   ```
6. Политика actions организации трогать не надо: workflow не использует ни одного action.
7. Settings → Actions → General → Workflow permissions: если пуш результатов падает с 403, включить «Read and write permissions». Если на `main` стоит защита ветки с обязательными PR, пуш результатов отклонится, тогда разрешить GitHub Actions обходить правило или поставить переменную `A2179_CI_NO_PUSH=1` (результаты останутся только в `ci_out/` на раннере).

## Переменные репы (Settings → Secrets and variables → Actions → Variables, всё необязательно)
| переменная | по умолчанию | что |
|---|---|---|
| `A2179_CI_DENSE` | `auto` | `auto` dense если есть, `on` то же, но причина отказа в отчёте, `off` без эмбеддера |
| `A2179_CI_VENV` | `~/.cache/a2179-ci/venv` | где venv |
| `A2179_CI_PYTHON` | `python3` | чем создать venv, например `python3.14` |
| `A2179_CI_ALLOW_DOWNLOAD` | не задано | `true` разрешает скачивать модель с Hugging Face |
| `A2179_CI_STEP_TIMEOUT` | `1800` | секунд на один шаг, весь job 180 минут |
| `HF_HOME` | `~/.cache/huggingface` | где кэш моделей |

Любая новая `A2179_CI_*` переменная доходит до `ci/run.sh` без правки YAML.

## Безопасность
- Триггер только `push` в `main` этой репы. Нет `pull_request`, `pull_request_target`, `workflow_run`, `schedule`, `workflow_dispatch`: код из PR и форков на раннере не исполняется. Плюс `if: github.repository == 'FreecodersInc/a2179_monitor'`.
- Код на твоей машине может запустить только тот, у кого есть push в `main`. Это и есть граница доверия.
- `permissions: contents: write`, больше ничего. Секретов кроме автоматического `GITHUB_TOKEN` нет, и он живёт только в окружении шага.
- Ни одного стороннего action: исполняется только код этой репы на пушнутом sha.
- Сеть: workflow ходит только в GitHub (fetch и push). pip ходит в PyPI, только когда поменялись requirements. Hugging Face только при `A2179_CI_ALLOW_DOWNLOAD=true`. Ничего никуда не отправляется.
- Без sudo, пишет только в чекаут и `$HOME` пользователя раннера.
- Один прогон за раз (`concurrency`), следующий ждёт в очереди, не отменяет текущий.

## Где результаты
`results/<UTC дата>_<short sha>.json` и рядом папка с отчётами, формат в [`results/README.md`](../results/README.md).
