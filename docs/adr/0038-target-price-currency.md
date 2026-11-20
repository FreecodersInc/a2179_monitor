---
status: accepted
date: 2026-11-20
deciders: Лёня, Grok Ebanutyi
commits: этот коммит
---

# ADR-0038: цена цели с валютой в профиле, target_usd как старое имя

## Контекст и проблема
Порог «ниже цели» был только в USD (`target_usd`), проверочные фразы эмбеддера были вшиты под A2179 (T-005).

## Решение
`target_price` {"amount", "currency"} (USD, EUR, RSD) в профиле или `config.json`, в USD по `rates`. Если `target_price` нет, работает `target_usd`. Явный `target_usd` в `config.json` или `A2179_TARGET_USD` перекрывает `target_price` профиля. Env `A2179_TARGET_PRICE`, `A2179_TARGET_CURRENCY`. Неизвестная валюта это ошибка. Проверочная пара эмбеддера из `probe` профиля, иначе первый позитивный и первый негативный прототип. Профиль A2179 называет свою старую пару и старую цену, поведение A2179 не меняется.

### Как проверяем
`tests/test_target_price.py`, замороженные дефолты A2179 (`tests/fixtures/a2179_defaults_frozen.json`) не тронуты.
