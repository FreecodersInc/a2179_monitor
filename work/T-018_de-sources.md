# T-018: источники вакансий: ToS, коллекторы API и RSS

| поле | значение |
|---|---|
| статус | in progress |
| майлстоун | M-003 |
| ответственный | Grok Ebanutyi, ревью Лёня |
| создана | 2026-11-19 |
| закрыта | |

## Цель
Коллекторы Remotive, Remote OK, Arbeitnow, We Work Remotely RSS, HN API, HelloWorld.rs через общий лимитер, с выводом по ToS у каждого.

## Критерии приёмки
- [x] ToS Remotive и Remote OK прочитаны, вывод ниже
- [ ] ToS и robots.txt Arbeitnow, We Work Remotely, HN API, HelloWorld.rs
- [x] Remotive не чаще 4 раз в день, кэш 24 часа, вакансии новее 24 часов отбрасываем, в отчёте ссылка назад и упоминание Remotive и Remote OK
- [x] офлайн тесты на сохранённых ответах, никакой сети в `tests/` (`tests/test_jobs.py`, `tests/test_review.py`)
- [x] LinkedIn, Indeed, Glassdoor не трогаем (код отказывается ходить на эти хосты)
- [ ] Arbeitnow, We Work Remotely RSS, HN API, HelloWorld.rs ещё не подключены

## План
1. По плану M-003, EXP запись до прогона.

## Ссылки
- гипотезы: H-02
- эксперименты: EXP-NNN при старте
- решения: ADR-0003
- коммиты:

## Заметки и итог
Минимальный путь уже в CI и в `python eval/review_run.py`.

### Remotive
Официальный JSON API `https://remotive.com/api/remote-jobs` (мы зовём `?category=data`). В ответе legal notice, проверено 2026-11-19: ссылка назад и упоминание Remotive, не перепубликовывать вакансии на сторонних досках, не чаще 4 запросов в день, вакансии могут быть с задержкой 24 часа. Делаем одно обращение в сутки, кэш 24 часа, дневной потолок 4 даже при `--fetch always`, свежие вакансии (младше 24 часов) в отчёт не берём. В `matches.md` / `all.md` только заголовок, компания, локация, короткий фрагмент и ссылка на Remotive.

### Remote OK
Официальный JSON API `https://remoteok.com/api`. В первом элементе ответа ToS: ссылка назад и упоминание Remote OK. Кэш 24 часа, в отчёте ссылка на карточку на Remote OK. Полные описания не перепубликуем.

### Ещё не делали
Arbeitnow, We Work Remotely RSS, HN API, HelloWorld.rs: ToS до кода ещё не закрыты, коллекторов нет.
