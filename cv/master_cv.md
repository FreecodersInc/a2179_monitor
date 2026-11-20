# Leonid Blokhin: Master CV

**Solution Architect · MLOps / Platform Engineer · Data Engineer**  
Novi Sad, Serbia · [leonid.blohin@gmail.com](mailto:leonid.blohin@gmail.com) · +381 (62) 110-6216 · +1 (925) 309-0952  
[LinkedIn](https://www.linkedin.com/in/lblokhin/) · [GitHub](https://github.com/leonid133) · Telegram @Leonid_blokhin

> Published with Leonid's consent (2026-11-20, ADR-0035). The most complete version: the master resume inventory, plus the links from his CVs and talks and press found and checked on the web. Short role versions (Data Engineer, AI Engineer, AIOps / LLMOps, Data Scientist, DevOps) are cut from this text.

## Professional Summary

Solution Architect and MLOps / Platform Engineer with 14+ years of professional experience across software development, data engineering, ML infrastructure and technical leadership. Progressed from Big Data Engineer to Development Team Lead at Provectus, with assignments including Nike, IMVU, NinthDecimal and Hydrosphere Mist. Combines Python/Scala data platforms, AWS and Kubernetes delivery, open-source feature-store integrations, and recent RAG/product engineering. Independent work spans private AI assistants, a Go/PostgreSQL platform, connectome graph matching and reproducible physiological-data research. This master resume is an extended career inventory: implementation details, context and outcome boundaries are retained for tailoring applications, interviews and project case studies.

## Technical and Delivery Skills

- **Programming:** Python, Go, Scala, C++, Bash; SQL for pipelines and relational data workflows.
- **Robotics and generative media:** UAV ground-station and Linux autopilot software, Navio2/Raspberry Pi integration, robot control/video monitoring, speech-driven avatar animation and SadTalker inference.
- **Python services:** FastAPI, Flask, Pydantic, SQLAlchemy, Alembic, Celery, pytest and Poetry.
- **Analysis:** pandas, NumPy, SciPy, scikit-learn, Jupyter, Matplotlib, Seaborn; EDA, descriptive PCA, regression, grouped validation, leakage investigation and reproducibility.
- **ML/AI platforms:** MLOps, Amazon SageMaker, Feast, feature stores, model-serving infrastructure, RAG, embeddings, semantic/vector search, LlamaIndex, Qdrant, sentence transformers and LLM service integration.
- **Data engineering:** Apache Spark, Airflow, ETL, Redshift, RDS, PostgreSQL, S3, DynamoDB; product analytics, application logs and geolocation data.
- **Cloud and delivery:** AWS, GCP, Azure, Linux, Docker/Compose, Kubernetes/EKS, Terraform, cloud networking, GitLab CI, Jenkins and Argo CD.
- **Observability:** OpenTelemetry, Prometheus, Grafana, Loki; logs, metrics, deployment troubleshooting and production support.
- **Architecture:** API contracts, background workflows, migration/rollback planning, technical discovery, estimates, roadmaps, statements of work, mentoring, hiring support and client presentations.
- **Additional implementation context:** Azure Blob Storage/AI Search and ASP.NET Core in CheckIT; River jobs and OpenAPI compatibility checks in the Go-based Vessyl backend.

## Professional Experience

### Chief Technology Officer / Solution Architect

**Spawn Solutions** · Oct 2025–Mar 2026  

Serbia

- Led technical discovery, architecture, estimation and delivery planning across client engagements and internal ventures; translated business requirements into statements of work and implementation plans.
- Coordinated backend, frontend and ML workstreams; supported hiring, task distribution, mentoring and delivery scope negotiations.
- Prepared demos and technical proposals and worked directly with clients, including on-site delivery collaboration for CheckIT.
- Supported AWS staging environments, security remediation and production troubleshooting; introduced structured project reporting and delivery metrics.

#### CheckIT: document processing and retrieval

- Delivered client-tested release candidates for CheckIT’s AI document-processing platform: semantic chunking, embeddings, vector retrieval, asynchronous callbacks, CI/CD and cloud deployment.
- Implemented ingestion and document update/delete workflows, token-aware semantic segmentation and tenant-scoped retrieval; integrated Azure Blob Storage, Azure AI Search and PostgreSQL in an ASP.NET Core service.
- Worked with idempotent ingestion identifiers, background indexing and a webhook outbox with bounded retries; made document lifecycle and callback states explicit.
- Completed the first delivery phase and prepared the next phase with the client; the recorded result is client-tested releases, without an asserted adoption or revenue metric.

#### HeraldAI: legal-tech MVP

- Defined RAG architecture, evaluation criteria and an implementation backlog for HeraldAI’s legaltech MVP; coordinated backend, frontend and ML delivery.
- Prepared the statement of work, roadmap and task breakdown; supported hiring, mentorship and scope management through two delivery sprints before the engagement ended.

#### Wafian and internal infrastructure

- Built and operated Wafian’s AWS environments and deployment pipelines; established repeatable releases and rollbacks and resolved production authentication issues.
- Prepared staging infrastructure, addressed AWS security issues and supported application and landing-page delivery.
- Reviewed Cappuccino, Wafian, Baristika and OrderWave during onboarding; prepared infrastructure for Baristika and contributed to cross-project technical coordination.

#### Technical discovery, presales and delivery recovery

- Analysed Promile’s existing system and prepared a two-year technical roadmap, complexity estimates and delivery options for commercial discussions.
- Prepared audits, estimates and statements of work for Ozdravlie and Build42; developed demos and technical cases for Finspot and Calea Digit. These were discovery/presales engagements rather than claimed production launches.
- Joined Colocio for a short delivery-recovery intervention, completed corrective work and helped stabilise a delayed client delivery.

### Independent Technology Consultant & Product Owner

**Self-employment** · 2024–2026  

Serbia

- Designed and implemented independent products and consulting solutions spanning private AI assistants, backend platforms, release automation and exploratory research.
- Combined requirements discovery, architecture, implementation, infrastructure, documentation and release planning in small-team product environments.
Individual implementations, research and concept-stage work are detailed in the project inventory below.

### Development Team Lead: MLOps Platform

**Provectus** · Feb 2021–Nov 2024  

Distributed / international client teams

- Led MLOps platform engineering at Provectus: team coordination, delivery planning, technical R&D and hands-on platform development.
- Promoted MLOps practices through internal demos, presentations, reports and knowledge sharing for engineering and business audiences.
- Worked with distributed teams and mentored junior engineers; balanced platform implementation, stakeholder communication and delivery priorities.

#### Nike: ML / MLOps engineering assignment

- Led MLOps/DevOps infrastructure development for a Nike client engagement and migrated ML workloads to Amazon SageMaker.
- Worked on infrastructure and developer workflows around managed ML execution, combining hands-on engineering with development-team coordination during November 2021–August 2022.

#### Talking-head virtual assistant: generative-media project

- Built a talking-head virtual-assistant workflow at Provectus that generated speech and video from a photograph and text, synchronising head motion, facial expressions and lip movements; experimented with multiple existing models.
- Ran SadTalker inference experiments, prepared speech audio and generated avatar demo clips; local working artefacts include an executed notebook, an audio-generation script and rendered videos.

### Big Data Engineer

**Provectus** · Feb 2016–Feb 2021  

Distributed / international client teams

- Built pipelines from RDS, Google Analytics, Mixpanel and application logs into Redshift; implemented ETL, reporting datasets and dashboards.
- Orchestrated data flows with Airflow and implemented monitoring and data-quality controls for transformations and reporting pipelines.
- Designed Spark processing jobs and improved CPU and memory utilisation for data-intensive analytics workloads.

#### IMVU: cloud migration (Dec 2019–Dec 2020)

- Contributed to IMVU’s migration from on-premises infrastructure to AWS, including data synchronisation, validation, drift detection and automated pipeline conversion.
- Worked on migration correctness and repeatability across legacy and cloud execution paths during the December 2019–December 2020 client assignment.

#### NinthDecimal: Senior Platform Engineer assignment (Mar 2017–Mar 2020)

- Developed geolocation analytics for NinthDecimal advertising reports and optimised Apache Spark jobs, CPU utilisation and memory consumption.
- For NinthDecimal, worked on a Python/Flask analytics scheduler with a configurable worker pool, execution states and HTTP callbacks; packaged dependencies with Docker and integrated tests into Jenkins.
- Supported the report-processing lifecycle and integration with persistent status services during the March 2017–March 2020 Senior Platform Engineer client assignment.

#### Hydrosphere.io: Software Engineer assignment (Feb 2016–Mar 2017)

- Core developer of Hydrosphere Mist, exposing Scala/Python Apache Spark jobs and ML models through service APIs.
- Worked on orchestration of Spark contexts, concurrent execution and failure recovery; integrated REST and messaging APIs and supported Python and Scala interoperability (February 2016–March 2017).
- Helped turn Spark workloads into callable services, bridging distributed data processing and application integration.

### Software Development

**Tensor** · 2014–2016  
- Worked in software development at Tensor during 2014–2016, alongside part-time autopilot development for Aeroxo.

### UAV Systems Engineering

**KNPP Vertolety MI+** · 2011–2014  
- Worked on UAV ground-station software and a propeller-balancing test stand at KNPP Vertolety MI+ during 2011–2014.

#### Additional startup and freelance software work

- Delivered backend services and integrations for startup and freelance engagements in fintech and HR technology; participated in MVP launches.
- Set up CI/CD, containerised applications, automated releases and monitoring for small engineering teams.

### Earlier Robotics and Control Projects

#### Aeroxo: Linux autopilot | Part-time, 2014–2016

- Developed Linux-based autopilot software for Aeroxo’s ERA x50-series tilt-rotor UAV platform using Navio2 and Raspberry Pi 3; work concerned the x50, x51 and x52 variants.
- Aeroxo’s letter dated 17 November 2016 acknowledged that the delivered software met the technical specification and solved the assigned task.

#### Mirrico: ROTC-TD tracked robot | Client acknowledgement

- Developed control and video-monitoring software for Mirrico’s ROTC-TD tracked robot, integrating two Laurent112 modules, an IP camera and a Windows 7 mini-PC.
- The client confirmed that the solution met requirements and had been used successfully for one year.
Aeroxo was part-time work alongside employment at Tensor. The Mirrico project date is not stated in the client letter.

## Independent Projects and Research

#### Macuo / Freecoders RAG platform ([macuo.globalnovatravel.com](https://macuo.globalnovatravel.com))

- Built Macuo, a private retrieval-augmented generation platform for domain-specific assistants, combining document ingestion, embeddings, vector retrieval and conversational interfaces.
- For Macuo, developed a modular Python RAG application with FastAPI, Celery workers, PostgreSQL/SQLAlchemy and Alembic migrations; separated API, worker, channel-adapter and storage concerns.
- Integrated LlamaIndex, Qdrant and sentence-transformer components; packaged application, data services and monitoring with Docker Compose.
- Developed Telegram-oriented assistant and memory workflows with per-chat prompts, contextual summaries and YouTrack knowledge/task integration.
- Used Prometheus metrics and Grafana/Loki in the application environment; maintained Python tests, formatting and dependency configuration.

#### Cash Flow Guardian: concept and pilot proposal

- Designed Cash Flow Guardian, a cash-flow and receivables monitoring concept, and prepared a pilot proposal; work covered the solution and prospective customer case rather than a completed commercial rollout.

#### Vessyl: product backend and delivery platform

- Architected a web/mobile platform with a Go/PostgreSQL backend, durable background workflows, observability and automated CI/CD; coordinated API contracts with web and mobile consumers.
- For Vessyl, developed the product API around an authored OpenAPI contract, generated types and compatibility checks against published and mobile-client snapshots.
- Worked on a separate API/worker runtime using PostgreSQL-backed River jobs and an outbox dispatcher; made retry budgets, deduplication and dead-letter handling explicit.
- Established versioned database migrations and release compatibility rules; kept schema changes compatible with the previous application release and checked transaction boundaries through integration tests.
- The backend is implemented in Go; the platform also includes web/mobile interfaces and content services. This does not imply authorship of every mobile or UI component.

#### idd74: CS-Cart release engineering

- Introduced version-controlled releases, database migrations and rollback procedures for the idd74 CS-Cart store.

#### Solar Landing: multilingual landing-page platform

- Built Solar Landing, a reusable multilingual landing-page platform with automated deployment.

#### FlyWire VNC Matching Challenge: connectome alignment (2025)

[Official challenge and leaderboard](https://codex.flywire.ai/app/vnc_matching_challenge)

- Developed graph-matching methods for the FlyWire VNC Matching Challenge and submitted a solution recorded on the official leaderboard in 2025.
- Engineered local graph features using bounded BFS neighbourhoods, degree and weight distributions, clustering and other subgraph statistics to compare connectome nodes.
- Explored seed-and-grow heuristics, local permutations and an embedding-to-nearest-neighbour candidate pipeline; evaluated Hungarian and min-cost max-flow approaches for one-to-one matching.
- The delivered result was a challenge submission and leaderboard entry. Graph-transformer training remained an idea in the working notes and is not claimed as a completed model.

#### Physiological-data research: H2S / HHcy programme

- Built a reproducible Python pipeline for physiological-data analysis, EDA and ML benchmarking, with explicit provenance, quality checks and grouped validation.
- Used pandas, NumPy, SciPy, scikit-learn and Jupyter for data preparation, visualisation, descriptive PCA, regression comparisons and residual diagnostics.
- In physiological-data research, compared ten regression families with a baseline using nested acquisition-batch-grouped validation; investigated leakage from related observations and outcomederived predictors.
- Documented a no-model selection result when candidates failed release criteria; produced model manifests and preserved earlier random-row experiments as unvalidated legacy artefacts.
- For physiological-data research, added automated checks, hashes and semantic reproducibility comparisons for generated artefacts; documented methods, data boundaries and permitted claims.
- Prepared prospective research gates and conditional decision economics for further work. The proposed experiment has not been run; clinical or therapeutic efficacy is not claimed.

#### Freecoders: reusable architecture and solution library

- Developed a Freecoders knowledge/solution library organising reusable architecture patterns, project evidence and implementation references across AI, data and infrastructure work.
- Distinguished source code, inferred architecture, concept briefs and intended outcome metrics; the catalogue is a reuse/discovery resource, not evidence that every listed solution was personally shipped.

#### Global Nova Travel, community and small prototypes

- Developed Global Nova Travel, organised local events in Serbia and pursued international partnerships; explored Telegram bots and small game prototypes alongside technical products.

## Open Source Contributions

#### Feast: AWS integration (2021)

- Contributed AWS provider support, a DynamoDB online feature store and an S3-backed registry to Feast, including integration tests; merged upstream.
- Worked on AWS configuration, registry/online-store integration, tests and examples; see Feast PR 1483 (2021).

#### SageMaker Studio notebook extension: AWS Samples (2021)

- Added scikit-learn notebook cell magics for submitting and managing SageMaker training jobs; merged into the AWS Samples SageMaker Studio notebook extension.
- Implemented the estimator integration, command arguments, extension registration and example notebook; see AWS Samples PR 2 (2021).
Hydrosphere Mist Core developer of Hydrosphere Mist; implementation scope is described in the Provectus assignment.

## Education and Professional Development

### Engineering degree: Thermal Power Engineering & Heat Technology

**Kazan State Technological University** · 2001–2008  
Postgraduate research and teaching, 2008–2009. The qualification is in energy/heat technology; software development was pursued alongside the engineering background. MIT Professional Education Applied Data Science Program. Professional education, not a degree in computer science.

## Languages

- English: fluent. Russian: proficient.

## Awards and Public Speaking

- Won 1st place in an AWS DeepRacer competition in San Francisco in 2022 and trained and tuned the reinforcement-learning model for autonomous racing. With that model he reached the world finals at AWS re:Invent in Las Vegas, where Provectus placed 7th of 192 participants ([LinkedIn post, Dec 2021](https://www.linkedin.com/posts/lblokhin_provectus-awsdeepracer-reinvent2021-activity-6873670041045491713-iVoP)).
- **2nd place, AWS DeepRacer League at Amazon re:MARS 2022 (Las Vegas)**, together with Rinat Gareev, after the NASA team. Racing models he developed were also used by teammates. Source: Provectus, [“Amazon re:MARS 2022: A Quick Recap and the Provectus Experience”](https://provectus.com/blog/amazon-remars-2022-recap/) (July 7, 2022).
- AWS DeepRacer credential: [credential.net](https://www.credential.net/36a719f7-0337-4670-b7f1-d38bd7af34ab).
- Talks on feature stores, Apache Spark, Parquet, Mist, neural networks in Python and Scala/Python interoperability, listed below. Recordings are in Russian unless marked English.

### Talks (recordings)
Playlist: [my conference presentations | мои доклады](https://www.youtube.com/playlist?list=PLyiZa9zMXnDUidEKCdAvL2t3s3rfO95HS)

| date | event | talk | links |
|---|---|---|---|
| 2021-09-20 | Saint HighLoad++ 2021, St Petersburg (with Dmitry Evstiukhin) | «Хранилище фич или какая-то дичь?» Feature stores in practice, Feast, AWS DynamoDB integration | [video](https://www.youtube.com/watch?v=YvRHhLAlJns) · [abstract](https://highload.ru/spb/2021/abstracts/7975) |
| 2017 (video 2017-10-03) | Data Science Meetup, Provectus (with Stepan Pushkarev), English | “Machine Learning in Spark” | [video](https://www.youtube.com/watch?v=6AXO9klzHcY) |
| 2017-05-20 | FrontDays 2017 | «Мист. Сервис для работы с Apache Spark» | [video](https://www.youtube.com/watch?v=UWHMJX4xlIQ) |
| 2017-04-22 | SECON 2017 | «Колоночные БД на примере Parquet»: row vs columnar storage, Parquet, Spark RDD/DataFrames/Datasets, Mist as a REST service | [video](https://www.youtube.com/watch?v=BxK69csWEb0) · [0x1.tv](https://0x1.tv/%D0%9A%D0%BE%D0%BB%D0%BE%D0%BD%D0%BE%D1%87%D0%BD%D1%8B%D0%B5_%D0%91%D0%94_%D0%BD%D0%B0_%D0%BF%D1%80%D0%B8%D0%BC%D0%B5%D1%80%D0%B5_Parquet_(%D0%9B%D0%B5%D0%BE%D0%BD%D0%B8%D0%B4_%D0%91%D0%BB%D0%BE%D1%85%D0%B8%D0%BD,_SECON-2017)) |
| 2016 (video 2016-12-15) | Python MeetUp, Expert Fridays | «Нейросети на питоне: пфф easy» | [video](https://www.youtube.com/watch?v=BQARnNsbL8U) |
| 2016 (video 2016-06-13) | Stachka 2016 | «Дружим языки Scala и Python» | [video](https://www.youtube.com/watch?v=2UaEfqs0_sA) |
| 2016 (video 2016-06-08) | Java Day Kazan 2016 | «Mist, как часть Hydrosphere» | [video](https://www.youtube.com/watch?v=OE-A7iFUYao) |
| 2016 (video 2016-04-12) | Expert Fridays | «Магия вуду над Spark Job» | [video](https://www.youtube.com/watch?v=HeiDYf0fKpw) |
| 2015-04-21 | VeloCPPed, C++ meetup, Tensor, Kazan | «Паттерны Factory и Variant, сильные и слабые стороны» | [video](https://www.youtube.com/watch?v=s8TQA-x8Qz0) |

Dates are the event dates from the event page or the video description, otherwise the video upload date. Two internal Provectus MLOps Platform demo recordings in the same playlist are left out here.

## Links

- LinkedIn: https://www.linkedin.com/in/lblokhin/
- GitHub: https://github.com/leonid133
- Feast PR 1483, AWS provider, DynamoDB online store, S3 registry (2021): https://github.com/feast-dev/feast/pull/1483
- AWS Samples SageMaker Studio notebook magic, scikit-learn training magic, PR 2 (2021): https://github.com/aws-samples/sagemaker-studio-notebook-magic-cell/pull/2
- Hydrosphere Mist (core developer): https://github.com/Hydrospheredata/mist
- FlyWire VNC Matching Challenge: https://codex.flywire.ai/app/vnc_matching_challenge
- Macuo, private RAG platform: https://macuo.globalnovatravel.com
- Provectus, re:MARS 2022 recap (DeepRacer 2nd place): https://provectus.com/blog/amazon-remars-2022-recap/
- AWS DeepRacer credential: https://www.credential.net/36a719f7-0337-4670-b7f1-d38bd7af34ab
- Talks playlist: https://www.youtube.com/playlist?list=PLyiZa9zMXnDUidEKCdAvL2t3s3rfO95HS
