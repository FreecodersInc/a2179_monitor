# EXP-005: matcher de_jobs на второй пачке (350 строк, все labeler = bot, 18 unsure)

| вид меток | P (95%) | пропущено в пуле, оценка (95%) | R оценка (95%) |
|---|---|---|---|
| E+S | 0.714 (0.648–0.771) | 1056 (835–1301) | 0.122 (0.102–0.15) |
| E | 0.403 (0.338–0.471) | 127 (58–269) | 0.396 (0.236–0.587) |
| E+S_unsure_as_pos | 0.723 (0.659–0.78) | 1077 (854–1323) | 0.122 (0.101–0.148) |
| E+S_unsure_as_neg | 0.655 (0.588–0.717) | 992 (777–1236) | 0.12 (0.098–0.148) |

| семейство заголовков | matcher верно | matcher ложно | пропущено (в выборке) | верно отсеяно |
|---|---|---|---|---|
| data engineer / analytics engineer | 79 | 1 | 0 | 0 |
| data scientist / analyst | 0 | 4 | 0 | 30 |
| machine learning / ai engineer | 17 | 8 | 19 | 11 |
| manager / director | 0 | 5 | 0 | 8 |
| other | 0 | 26 | 1 | 19 |
| platform / infrastructure / cloud | 31 | 15 | 18 | 26 |
| site reliability / devops | 9 | 0 | 6 | 0 |
| software engineer + data platform/infra | 4 | 0 | 5 | 0 |
| solutions architect | 7 | 0 | 1 | 0 |
