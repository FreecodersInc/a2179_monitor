# EXP-000: нативный крэш эмбеддера (WSL2 + CUDA torch)

| поле | значение |
|---|---|
| статус | waiting, fix shipped, ждём подтверждения на машине, где падало |
| дата | 2026-11-19 |
| тип | инженерное расследование (не метрика качества) |
| код (commit) | a9fb3cd (изоляция в дочернем процессе), d8cc345 (cpu guard) |
| данные | n/a |
| окружение, где падало | WSL2, Python 3.14.4, torch 2.15.0+cu132, CUDA 13.2, sentence-transformers 5.7.0, transformers 5.23.0 |
| окружение тестов | Linux, Python 3.13.5, без torch (поддельные модули и дочерние процессы, которые сами себя роняют) |
| seed | детерминировано |
| артефакты | `tests/test_embed_probe.py`, README «Эмбеддер не работает» |

## Вопрос и гипотеза
Почему `python monitor.py --check-embedder` падает с segfault и как сделать, чтобы падение библиотеки никогда не роняло основной процесс.
Гипотеза 1: нативный крэш внутри torch/CUDA, Python исключением его не поймать, значит нужна изоляция процессом.
Гипотеза 2 (после стека): на `device=cpu` CUDA трогать вообще не должны, значит кто то её всё равно инициализирует.

## Условия и установка
Симптом на машине Лёни. Стек faulthandler (его прогон): `SentenceTransformer('all-MiniLM-L6-v2', device='cpu').encode(...)` падает в `transformers/utils/import_utils.py:1916 is_cuda_stream_capturing` → `torch/cuda/graphs.py:92 is_current_stream_capturing`. При этом `torch.ones(3) @ torch.ones(3)` работает.

## Процедура
1. Загрузка модели и тестовый encode вынесены в дочерний процесс (`python -X faulthandler -m a2179.embed_probe`). Сигнал (SIGSEGV, SIGABRT), ненулевой выход без результата или таймаут → стадия `crashed` / `probe_failed` / `probe_timeout`, откат на regex + TF-IDF, torch в основной процесс не грузится.
2. Порядок импорта: torch раньше sentence-transformers (и sklearn).
3. Устройство из конфига `embedder.device` (auto | cuda | cpu), проверка отчитывается, на каком устройстве прошла.
4. Для cpu: `CUDA_VISIBLE_DEVICES=""` до импорта torch, `torch.cuda.is_current_stream_capturing` и `transformers.utils.import_utils.is_cuda_stream_capturing` подменяются на False до загрузки модели (цель патча проверена по исходнику transformers v5.23.0).
```bash
python -m unittest tests.test_embed_probe -v
python monitor.py --check-embedder
```

## Результаты
- В тестах дочерний процесс, который убивает себя SIGSEGV или `os.abort()`, и зависший процесс: монитор каждый раз доходит до конца прохода на regex + TF-IDF, модель в основном процессе не создаётся. 29 тестов в `tests/test_embed_probe.py`, все зелёные в окружении тестов.
- На машине, где падало, исправление (d8cc345) ещё не подтверждено отдельным прогоном `--check-embedder`. Его прогон eval в 17:20 CET (EXP-001) с MiniLM прошёл, но в логе не записано окружение.

## Выводы
- Гипотеза 1 подтверждена как класс проблемы: падение нативной библиотеки ловится только границей процесса.
- Гипотеза 2 подтверждена стеком: transformers 5.x спрашивает CUDA (graph capture check) даже на cpu, а инициализация CUDA под WSL2 падает.

## Угрозы валидности
Настоящий torch и CUDA в тестах не запускались, крэш имитирован. Падение, которое случится позже на конкретных текстах (уже после успешной проверки), изоляция не ловит, encode в основном процессе.

## Следующие шаги
Прогон `python monitor.py --check-embedder` на машине Лёни с `"device": "cpu"` и `"device": "cuda"`, записать сюда вывод. Для cuda проверить драйвер WSL: `nvidia-smi`, `python -c "import torch; torch.zeros(1).cuda()"`.
