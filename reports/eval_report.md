# Eval report (2026-11-19 17:20 CET)

Set: 99 items, 42 positive, 57 negative. Sources: {'kp_live_2026-11-19': 26, 'shop_live_2026-11-19': 12, 'synthetic': 59, 'kp_dryrun_2026-11-19': 2}. Languages: {'sr': 76, 'en': 10, 'ru': 9, 'sr_cyr': 4}.

Embedder: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

| mode | precision | recall | F1 | TP | FP | FN | TN | thresholds | 2-fold P/R/F1 |
|---|---|---|---|---|---|---|---|---|---|
| regex | 0.976 | 0.976 | 0.976 | 41 | 1 | 1 | 56 |  |  |
| tfidf | 0.808 | 1.000 | 0.894 | 42 | 10 | 0 | 47 | {'pos': 0.2, 'margin': 0.02} | 0.830/0.929/0.876 |
| hybrid_tfidf | 0.977 | 1.000 | 0.988 | 42 | 1 | 0 | 56 | {'pos': 0.38, 'margin': 0.18} | 0.976/0.976/0.976 |
| embed | 0.600 | 1.000 | 0.750 | 42 | 28 | 0 | 29 | {'pos': 0.66, 'margin': -0.06} | 0.615/0.952/0.748 |
| hybrid_embed | 0.840 | 1.000 | 0.913 | 42 | 8 | 0 | 49 | {'pos': 0.84, 'margin': 0.0} | 0.911/0.976/0.943 |

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

## Errors: embed (28)
- id 2 label=0: Apple MacBook Air A2179 (2020) – Core i3 / 8GB RAM / 256GB S  (0.729, 0.767)
- id 4 label=0: Srafovi MacBook Air a2179 (2020)  (0.891, 0.91)
- id 5 label=0: Konektori MacBook Air a2179 (2020)  (0.876, 0.901)
- id 6 label=0: Kuler MacBook Air a2179 (2020)  (0.864, 0.896)
- id 7 label=0: WiFi antene MacBook Air a2179 (2020)  (0.712, 0.728)
- id 8 label=0: Donji poklopac MacBook Air a2179 (2020)  (0.911, 0.92)
- id 9 label=0: Tastatura sa palmrest MacBook Air a2179 (2020)  (0.86, 0.869)
- id 13 label=0: UK zastita za tastaturu za MacBook Air 13" M1 A2337/A2179  (0.705, 0.761)
- id 16 label=0: Cover za Macbook Air 13 2018/2020 (A2337 M1 A2179 A1932) crn  (0.805, 0.813)
- id 19 label=0: Zamena ekrana za MacBook Air 2018-2020 A1932 / A2179 ORG  (0.815, 0.825)
- id 22 label=0: Cover za Macbook Air 13 2018/2020 (A2337 M1 A2179 A1932) crni  (0.846, 0.871)
- id 30 label=0: Zastita za ekran za MacBook Air 13″ (2018-2021) A1932 A2179 A2337 M1  (0.932, 0.933)
- id 35 label=0: Flet kabl do ploce za audio i power dugme MacBook Air Retina 13″ A1932 | A2179  (0.8, 0.796)
- id 74 label=0: MacBook Air 2020 i5 16GB 512GB, ekran bez ogrebotina  (0.843, 0.834)
- id 75 label=0: MacBook Air 2020 i3 sa slomljenim ekranom, ostalo radi  (0.877, 0.843)
- id 77 label=0: Punjač 30W za MacBook Air 2020  (0.8, 0.837)
- id 79 label=0: Kaljeno staklo za ekran MacBook Air A2179  (0.851, 0.842)
- id 81 label=0: Зарядка для MacBook Air A2179 30W  (0.894, 0.951)
- id 83 label=0: Тастатура за MacBook Air A2179  (0.935, 0.966)
- id 87 label=0: Kupujem MacBook Air A2179 u bilo kom stanju  (0.858, 0.905)
- id 88 label=0: Tražim ekran za MacBook Air A2179  (0.945, 0.915)
- id 89 label=0: Servis za MacBook: zamena LCD ekrana A2179 za 1h  (0.901, 0.895)
- id 90 label=0: Kamera FaceTime HD za MacBook Air A2179  (0.786, 0.717)
- id 92 label=0: MacBook Air 13 2017 A1466 ekran  (0.945, 0.934)
- id 93 label=0: Šarke za ekran MacBook Air A2179  (0.959, 0.964)
- id 94 label=0: Lepak traka za LCD MacBook Air A2179  (0.917, 0.915)
- id 95 label=0: Folija za ekran MacBook Air 13  (0.904, 0.935)
- id 96 label=0: MacBook Air A2179 ne pali, ekran razbijen  (0.889, 0.868)

## Errors: hybrid_embed (8)
- id 74 label=0: MacBook Air 2020 i5 16GB 512GB, ekran bez ogrebotina  (0.843, 0.834)
- id 75 label=0: MacBook Air 2020 i3 sa slomljenim ekranom, ostalo radi  (0.877, 0.843)
- id 79 label=0: Kaljeno staklo za ekran MacBook Air A2179  (0.851, 0.842)
- id 88 label=0: Tražim ekran za MacBook Air A2179  (0.945, 0.915)
- id 89 label=0: Servis za MacBook: zamena LCD ekrana A2179 za 1h  (0.901, 0.895)
- id 92 label=0: MacBook Air 13 2017 A1466 ekran  (0.945, 0.934)
- id 94 label=0: Lepak traka za LCD MacBook Air A2179  (0.917, 0.915)
- id 96 label=0: MacBook Air A2179 ne pali, ekran razbijen  (0.889, 0.868)
