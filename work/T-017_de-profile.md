# T-017: профиль цели Data Engineer из утверждённых навыков

| поле | значение |
|---|---|
| статус | in progress |
| майлстоун | M-003 |
| ответственный | Grok Ebanutyi, ревью Лёня |
| создана | 2026-11-19 |
| закрыта | |

## Цель
`targets/de_jobs.json`: титулы, соседние роли как трудные негативы, навыки как группы признаков, seniority, локация, языки, зарплата, прототипы. Только то, что Лёня утвердил.

Пока публичный `targets/de_jobs.json` не заведён. Утверждённый профиль для матчинга лежит локально в `targets/de_jobs.local.json` (gitignore). География матчинга: Канада (Toronto, Vancouver, remote Canada) и EU remote на английском.

## Критерии приёмки
- [x] Лёня прислал CV в личку, CV не в репо (`private/` в `.gitignore`)
- [x] список навыков и предпочтений, утверждённый Лёней, в задаче
- [ ] публичный `targets/de_jobs.json` проходит `tests/test_target.py` (компиляция, не пустые прототипы)
- [x] в профиле нет имени, контактов, работодателей, дат

## Утверждённый список (без личных данных)

- Роли в базе для матчинга (6): Data Engineer, AI Engineer, AIOps / LLMOps Engineer, Data Scientist, DevOps Engineer, плюс общий master CV. Для Data Engineer есть два варианта CV.
- Seniority: 14+ лет, уровень lead / architect.
- Локация: сейчас Novi Sad, Serbia. Приоритет релокация в Канаду (Toronto, Vancouver). Remote ок, плюс EU remote на английском.
- Языки: English fluent, Russian native. Сербский не указываем.
- Добавлено по утверждению: Kafka, dbt, Snowflake, Databricks.

Навыки группами:
- Обработка и языки: Apache Spark, Scala, Python, SQL, Bash, Go, C++, pandas, NumPy
- Пайплайны и хранение: Airflow, ETL, Redshift, RDS, PostgreSQL, S3, DynamoDB, Kafka, dbt, Snowflake, Databricks
- Сервисы и надёжность: Flask, FastAPI, Celery, SQLAlchemy, Alembic, pytest, Pydantic
- Облако и поставка: AWS, GCP, Azure, Linux, Docker, Kubernetes / EKS, Terraform, Jenkins, GitLab CI, Argo CD
- Наблюдаемость: Prometheus, Grafana, Loki, OpenTelemetry
- ML / AI: Amazon SageMaker, Feast, feature stores, RAG, embeddings, LlamaIndex, Qdrant, sentence transformers, scikit-learn, SciPy

Веса:
- Обязательные: SQL, Python или Scala, плюс один из Spark / Airflow / ETL / dbt
- Плюсы: Kafka, dbt, Snowflake, Databricks, облако, Redshift / PostgreSQL / S3, Docker / Kubernetes, Terraform, data quality, SageMaker, Feast, RAG
- Не цель: только Excel, только BI отчёты, курсы и буткемпы, неоплачиваемые стажировки

Зарплата в CV и в утверждениях не указана, порог не ставим.

## План
1. По плану M-003, EXP запись до прогона.
2. Сбор английских вакансий (Канада и EU remote) и матчинг по локальному профилю.
3. Когда список стабилен, перенести в публичный `targets/de_jobs.json` без личных данных.

## Ссылки
- гипотезы: H-02
- эксперименты: EXP-NNN при старте
- решения: ADR-0010
- коммиты:

## Заметки и итог
Локальный профиль: `targets/de_jobs.local.json`, копии CV в `private/` (оба пути в gitignore). Компилируется тем же загрузчиком целей, 15 положительных и 15 отрицательных прототипов, без имени, контактов, работодателей и дат. Публичный `targets/de_jobs.json` ещё не коммитился, поэтому задача не закрыта.
