# CI на своём раннере (эксперименты на CUDA)

Workflow лежит в [`ci/experiments.yml`](../ci/experiments.yml). Решение и причины: [ADR-0008](adr/0008-ci-on-self-hosted-runner.md).

## Включить (один раз, на твоей стороне)
Токен, которым пушит Telega, не может писать в `.github/workflows`, поэтому файл лежит в `ci/` и GitHub его пока не видит. Включаешь сам, одной командой из корня репы:
```
mkdir -p .github/workflows && git mv ci/experiments.yml .github/workflows/ && git commit -m 'ci: enable experiments workflow' && git push
```
После этого workflow живёт в `.github/workflows/experiments.yml`. Правки в нём дальше тоже делаешь ты (или токен с правом Workflows).

## Что он делает
На каждый push в `main` этой репы (и только на него):
1. checkout
2. `ci/setup_venv.sh`: постоянный venv вне чекаута (`~/.cache/a2179-ci/venv`), `pip install -r requirements.txt` только когда файл поменялся, `requirements-embed.txt` только если в venv уже стоит torch
3. `ci/run_experiments.py`: каждый шаг в своём процессе, с кодом выхода, временем и хвостом вывода
   - `ci/env_probe.py`: python, torch, CUDA, имя GPU
   - unit тесты
   - `monitor.py --check-embedder` (модель сначала грузится в дочернем процессе, ADR-0005)
   - `eval/run_eval.py` (с эмбеддером, если проверка прошла, иначе `--no-embed`)
   - `eval/sweep.py --compute-dense` (иначе без dense)
   - если dense шаг всё равно упал, он перезапускается без эмбеддера и помечается `fallback`
4. `ci/write_results.py`: пишет `results/<UTC дата>_<short sha>.json`, коммитит только этот файл с `[skip ci]` и пушит в `main` (при отказе `git pull --rebase` и ещё раз, до 5 попыток)
5. тот же JSON и папка `ci_out/` (полные отчёты eval и sweep) уходят артефактом `experiments-<sha>-<attempt>` на 30 дней

Петли нет: `results/**` в `paths-ignore`, в сообщении `[skip ci]`, и push через `GITHUB_TOKEN` вообще не запускает workflow. Job падает красным, только если упали unit тесты, результаты при этом всё равно пишутся.

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
6. Settings → Actions → General → Workflow permissions: если пуш результатов падает с 403, включить «Read and write permissions». Если на `main` стоит защита ветки с обязательными PR, пуш результатов отклонится, тогда разрешить GitHub Actions обходить правило или убрать запись результатов коммитом (артефакт останется).

## Переменные репы (Settings → Secrets and variables → Actions → Variables, всё необязательно)
| переменная | по умолчанию | что |
|---|---|---|
| `A2179_CI_DENSE` | `auto` | `auto` dense если есть, `on` то же, но причина отказа в отчёте, `off` без эмбеддера |
| `A2179_CI_VENV` | `~/.cache/a2179-ci/venv` | где venv |
| `A2179_CI_PYTHON` | `python3` | чем создать venv, например `python3.14` |
| `A2179_CI_ALLOW_DOWNLOAD` | не задано | `true` разрешает скачивать модель с Hugging Face |
| `A2179_CI_STEP_TIMEOUT` | `1800` | секунд на один шаг, весь job 120 минут |

## Безопасность
- Триггер только `push` в `main` этой репы. Нет `pull_request`, `pull_request_target`, `workflow_run`, `schedule`, `workflow_dispatch`: код из PR и форков на раннере не исполняется. Плюс `if: github.repository == 'FreecodersInc/a2179_monitor'`.
- Код на твоей машине может запустить только тот, у кого есть push в `main`. Это и есть граница доверия.
- `permissions: contents: write`, больше ничего. Секретов кроме автоматического `GITHUB_TOKEN` нет.
- Сеть: workflow ходит только в GitHub (checkout, push, артефакт). pip ходит в PyPI, только когда поменялись requirements. Hugging Face только при `A2179_CI_ALLOW_DOWNLOAD=true`. Ничего никуда не отправляется.
- Без sudo, пишет только в чекаут и `$HOME` пользователя раннера.
- Один прогон за раз (`concurrency`), следующий ждёт в очереди, не отменяет текущий.

## Где результаты
`results/<UTC дата>_<short sha>.json`, формат в [`results/README.md`](../results/README.md). Полные отчёты в артефакте прогона.
