# Eval report: DE jobs (2026-11-20 00:11 CET)

Set: 112 items, 10 positive, 102 negative. Sources: {'real:remotive': 15, 'real:remoteok': 97}. Languages: {'en': 112}.

Embedder: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

| mode | precision | recall | F1 | TP | FP | FN | TN | thresholds | 2-fold P/R/F1 |
|---|---|---|---|---|---|---|---|---|---|
| regex | 1.000 | 0.200 | 0.333 | 2 | 0 | 8 | 102 |  |  |
| tfidf | no threshold reaches recall 1.0 |||||||||
| hybrid_tfidf | no threshold reaches recall 1.0 |||||||||
| embed | no threshold reaches recall 1.0 |||||||||
| hybrid_embed | no threshold reaches recall 1.0 |||||||||

Confusion: TP/FP/FN/TN above (positive = a match for DE jobs: English-language Data Engineer vacancies. Geography: Canada (Toronto, Vancouver, remote Canada) and EU remote in English. Public matching profile, no personal data.).

## Errors: regex (8)
- id 4 label=1: Senior DevOps Engineer  (model=devops; no part term)
- id 5 label=1: AWS Cloud Solutions Architect  (model=none; no matching engineering or data role)
- id 19 label=1: Cloud Engineer Automation  (model=devops; no part term)
- id 22 label=1: AI ML Engineer  (model=none; no matching engineering or data role)
- id 29 label=1: Senior Agentic AI Engineer NDA  (model=ai_platform; no part term)
- id 33 label=1: Platform and Integration Engineer Security Telemetry  (model=none; no matching engineering or data role)
- id 42 label=1: AI Engineer Data APIs  (model=ai_platform; no part term)
- id 92 label=1: Feltsense AI Engineer  (model=ai_platform; no part term)
