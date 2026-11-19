# Eval report (2026-11-19 10:02 UTC)

Set: 97 items, 40 positive, 57 negative. Sources: {'kp_live_2026-11-19': 26, 'shop_live_2026-11-19': 12, 'synthetic': 59}. Languages: {'sr': 74, 'en': 10, 'ru': 9, 'sr_cyr': 4}.

Embedder: unavailable (not_installed): ModuleNotFoundError: No module named 'sentence_transformers'

| mode | precision | recall | F1 | TP | FP | FN | TN | thresholds | 2-fold P/R/F1 |
|---|---|---|---|---|---|---|---|---|---|
| regex | 0.975 | 0.975 | 0.975 | 39 | 1 | 1 | 56 |  |  |
| tfidf | 0.800 | 1.000 | 0.889 | 40 | 10 | 0 | 47 | {'pos': 0.2, 'margin': 0.02} | 0.822/0.925/0.871 |
| hybrid_tfidf | 0.976 | 1.000 | 0.988 | 40 | 1 | 0 | 56 | {'pos': 0.38, 'margin': 0.18} | 0.975/0.975/0.975 |

Confusion: TP/FP/FN/TN above (positive = A2179-compatible display, lid or donor offer).

## Errors: regex (2)
- id 62 label=1: Prodajem LCD od Macbook Air 2020 (Intel i5), ispravan  (model=weak; laptop_specs=i5)
- id 96 label=0: MacBook Air A2179 ne pali, ekran razbijen  (model=strong; donor=ne pali)

## Errors: tfidf (10)
- id 5 label=0: Konektori MacBook Air a2179 (2020)  (0.251, 0.163)
- id 7 label=0: WiFi antene MacBook Air a2179 (2020)  (0.249, 0.161)
- id 75 label=0: MacBook Air 2020 i3 sa slomljenim ekranom, ostalo radi  (0.302, 0.159)
- id 79 label=0: Kaljeno staklo za ekran MacBook Air A2179  (0.27, 0.184)
- id 88 label=0: Tražim ekran za MacBook Air A2179  (0.37, 0.252)
- id 90 label=0: Kamera FaceTime HD za MacBook Air A2179  (0.212, 0.114)
- id 92 label=0: MacBook Air 13 2017 A1466 ekran  (0.291, 0.202)
- id 93 label=0: Šarke za ekran MacBook Air A2179  (0.399, 0.272)
- id 94 label=0: Lepak traka za LCD MacBook Air A2179  (0.282, 0.188)
- id 96 label=0: MacBook Air A2179 ne pali, ekran razbijen  (0.237, 0.156)

## Errors: hybrid_tfidf (1)
- id 96 label=0: MacBook Air A2179 ne pali, ekran razbijen  (0.237, 0.156)
