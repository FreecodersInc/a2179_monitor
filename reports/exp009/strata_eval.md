# EXP-005: matcher de_jobs на второй пачке (350 строк, все labeler = bot, 18 unsure)

| вид меток | P (95%) | пропущено в пуле, оценка (95%) | R оценка (95%) |
|---|---|---|---|
| E+S | 0.728 (0.664–0.784) | 1119 (894–1366) | 0.118 (0.099–0.144) |
| E | 0.403 (0.338–0.471) | 127 (58–269) | 0.396 (0.236–0.587) |
| E+S_unsure_as_pos | 0.738 (0.674–0.793) | 1140 (914–1387) | 0.118 (0.099–0.143) |
| E+S_unsure_as_neg | 0.67 (0.603–0.73) | 1056 (835–1301) | 0.116 (0.096–0.142) |

| семейство заголовков | matcher верно | matcher ложно | пропущено (в выборке) | верно отсеяно |
|---|---|---|---|---|
| data engineer / analytics engineer | 79 | 1 | 0 | 0 |
| data scientist / analyst | 0 | 4 | 0 | 30 |
| machine learning / ai engineer | 18 | 7 | 22 | 8 |
| manager / director | 0 | 5 | 0 | 8 |
| other | 2 | 24 | 1 | 19 |
| platform / infrastructure / cloud | 31 | 15 | 18 | 26 |
| site reliability / devops | 9 | 0 | 6 | 0 |
| software engineer + data platform/infra | 4 | 0 | 5 | 0 |
| solutions architect | 7 | 0 | 1 | 0 |

Правила профиля, перезапущенные сейчас (Horvitz-Thompson по стратам, bootstrap 95%):

| метки | P (95%) | R (95%) |
|---|---|---|
| E+S | 0.95 (0.849–0.998) | 0.363 (0.261–0.476) |
| E | 0.163 (0.11–0.241) | 0.377 (0.24–0.668) |
