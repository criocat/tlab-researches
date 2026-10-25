# Остановка RL по validation и контроль дополнительных SFT шагов

Фиксированные 200 updates могли быть слишком длинным дообучением. До новых результатов заданы шаги 20/50/100/150/200/400 и два метода: Monte Carlo с четырьмя действиями и exact expectation по 100 кандидатам frozen reference. Для каждого три matched seed 42/43/44. Reference и Context RM фиксированы.

Learning rate 3e-5, KL=0,5, CE=0,2, batch=16 и исходные 100k fine-tuning контекстов сохранены. Каждому методу и continued-SFT контролю доступны одинаковые minibatches и кеш reward. Контроль имеет нулевой reward gradient и не участвует в выборе RL.

## Выбор до test

Для 12 method/step вариантов максимизируется средний validation watch-proxy при NDCG и Recall не ниже 99% reference и max-item fraction не выше reference+0,02. Reference допускается как fallback. Test оценивает только reference, выбранный RL и continued-SFT на том же шаге. Никакой момент остановки по test не выбирается.

Выбран **exact, 50 updates**. Это выбор настройки по средним трёх seed, а не выбор самого удачного seed. Числа ниже — средние метрик, не качество ансамбля logits.

## Validation path

| method | step | ndcg@10 | recall@10 | context_rm_watch@10 | max_item_user_fraction |
| --- | --- | --- | --- | --- | --- |
| continued_sft | 20 | 0.134846 | 0.222833 | 0.435738 | 0.701333 |
| continued_sft | 50 | 0.134910 | 0.222833 | 0.436316 | 0.699167 |
| continued_sft | 100 | 0.134240 | 0.222500 | 0.436583 | 0.697667 |
| continued_sft | 150 | 0.133832 | 0.223167 | 0.436713 | 0.694500 |
| continued_sft | 200 | 0.134502 | 0.223333 | 0.436728 | 0.694667 |
| continued_sft | 400 | 0.134708 | 0.224333 | 0.436864 | 0.694333 |
| exact | 20 | 0.134860 | 0.222167 | 0.439001 | 0.703000 |
| exact | 50 | 0.134529 | 0.223000 | 0.444114 | 0.702833 |
| exact | 100 | 0.134192 | 0.223000 | 0.451741 | 0.704000 |
| exact | 150 | 0.133386 | 0.221667 | 0.456623 | 0.706500 |
| exact | 200 | 0.132548 | 0.219500 | 0.460748 | 0.711500 |
| exact | 400 | 0.130991 | 0.219333 | 0.473717 | 0.728667 |
| monte_carlo | 20 | 0.135091 | 0.222667 | 0.437877 | 0.702333 |
| monte_carlo | 50 | 0.134945 | 0.223333 | 0.442314 | 0.702333 |
| monte_carlo | 100 | 0.134276 | 0.222833 | 0.448756 | 0.703000 |
| monte_carlo | 150 | 0.133727 | 0.221667 | 0.453878 | 0.704333 |
| monte_carlo | 200 | 0.133102 | 0.220667 | 0.457122 | 0.706333 |
| monte_carlo | 400 | 0.131313 | 0.218333 | 0.469022 | 0.720500 |
| reference | 0 | 0.135639 | 0.223500 | 0.435525 | 0.702500 |

## Test после фиксации настройки

| method | ndcg@10 | recall@10 | context_rm_watch@10 | max_item_user_fraction |
| --- | --- | --- | --- | --- |
| continued_sft | 0.117156 | 0.194833 | 0.426456 | 0.691667 |
| exact | 0.117206 | 0.195000 | 0.434011 | 0.704500 |
| reference | 0.116802 | 0.194000 | 0.425677 | 0.695000 |

| comparison | delta NDCG | CI low | CI high |
| --- | --- | --- | --- |
| exact | 0.000404 | -0.001187 | 0.001956 |
| continued_sft | 0.000354 | -0.000578 | 0.001243 |
| selected_rl_minus_matched_control | 0.000050 | -0.001300 | 0.001407 |

Короткая RL-политика повышает predicted reward, сохраняя ordinary relevance в пределах точности этого эксперимента. Ordinary NDCG почти столь же хорошо меняется у matched continued-SFT. Поэтому точечный прирост точности нельзя целиком приписать reward gradient; CI к reference и контролю пересекают ноль.

## Проверка воспроизводимости и пределы

На шаге 200 все шесть RL checkpoints бит-в-бит совпали с прежними fixed200 моделями. Дополнительные validation scores не изменили обучение; prefix training indices и top100 support также совпали по SHA. Это соединяет эксперименты общей траекторией, без незаметной смены оптимизатора.

Выбор JSON сохранён и захеширован до test. Пользовательские CI условны на фиксированных SFT/RM, дополнительных настройках и уже исследованном KION. Они не учитывают uncertainty выбора по reused validation и не доказывают причинного или онлайн-эффекта. Старые fixed200 fallback результаты сохранены.

Воспроизведение: `python -m task2.validation_path` после full-data/Context RM/conditional full-reference этапов. Полные protocol, validation curves, per-user CSV, prefix identity и checksum выбора находятся в `artifacts/task2_validation_path/`.
