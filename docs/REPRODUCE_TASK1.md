# Воспроизведение задания 1

[Установка и данные](ENVIRONMENT.md) · [Отчёт](../task1/report.pdf).

[Сохранённые локальные результаты](../result_artifacts/task1/metrics.csv) · [Протокол](../result_artifacts/task1/protocol.json).

## Один основной запуск с seed 42

```sh
python reproduce.py task1 --dry-run
python reproduce.py task1
```

Скрипт загружает проверенный KION, строит признаки и кандидатов, обучает
ручной CatBoost и варианты автоматического поиска окон. Затем воспроизводит
поиск выражений по сохранённым random/LLM предложениям. Новая генерация LLM
не нужна. NumPy/Python/CatBoost seed — 42, hash seed дочерних процессов — 42.

Выходы сохраняются отдельно:

| Файл | Содержание |
| --- | --- |
| `artifacts/reproduced/task1/metrics.csv` | Manual, окна и абляции для seed 42 |
| `artifacts/reproduced/task1/selection.json` | Выбор окон только по validation |
| `artifacts/reproduced/task1_promptfe/metrics.csv` | Manual, random и LLM-поиск |
| `artifacts/reproduced/task1_promptfe/proposals.json` | Решения по каждому предложению |
| `artifacts/reproduced/task1_results.md` | Таблицы нового запуска |

Ориентиры исходного CPU seed 42: manual NDCG@10 ≈0,458863,
auto-selected ≈0,464198; LLM-поиск ≈0,459765, random ≈0,461318.
Средние трёх seed в PDF — другие числа: manual ≈0,461065 и auto ≈0,464169.
Сравнивать нужно соответствующие строки и одинаковые кандидаты.

Временной протокол: snapshots 12/19/26 июля и 2 августа; validation 9–15 августа;
test 16–22 августа. Все агрегаты используют только `timestamp < cutoff`.
Негативы синтетические, просмотренное исключается; это не CTR или полный retrieval.

## Полное повторение дополнительных таблиц

Этот раздел опционален: он воспроизводит средние нескольких seed, ранние недели,
графовые признаки, расширение истории и полный каталог. Для единственного
основного запуска достаточно предыдущего раздела.

Выполнить по порядку в свежем клоне:


```sh
python -m shared.download
python -m task1.run --users 50000 --queries 1500 --iterations 350 --seeds 42 43 44
python -m task1.promptfe_run --seed 42 --syntax infix --proposals 12 --iterations 350 --replay replay/feature_search_cpu.json
python -m shared.analyze --task task1
python -m task1.temporal_study
python -m task1.relational
python -m task1.relational_temporal
python -m task1.ranking_study
python -m task1.population_study
python -m task1.cohort_study
python -m task1.full_catalog
python -m task1.catalog_calibration
python -m task1.full_catalog --queries 512 --sampling-seed 20261027 --exclude-prior --calibration artifacts/task1_catalog_calibration/selection.json --out artifacts/task1_full_catalog_replication
python -m task1.rolling_catalog
python -m task1.decayed_graph
```

Отдельный strict-RPN контроль воспроизводится сохранёнными ответами:

```sh
python -m task1.promptfe_run --seed 42 --syntax rpn --proposals 12 --iterations 350 --replay replay/feature_search_rpn_cpu.json --out artifacts/task1_promptfe_rpn
```

GPU infix-таблица использовала NumPy 2.1.3, Pandas 2.3.3, CatBoost 1.2.10,
scikit-learn 1.6.1; версии Torch — 2.11.0+cu128, Transformers — 4.57.6. Для неё сначала
построить базовые признаки с этими версиями, затем использовать
`--replay replay/feature_search_gpu.json --out artifacts/task1_gpu_replay`.
Для строгого GPU RPN есть `replay/feature_search_rpn_gpu.json`.
Это отдельные proposal sets; CPU и GPU строки не смешиваются.

Команды используют предусмотренные внутри протоколов seed 42/43/44,
одинаковые candidate sets и выбор только по validation. Популяционные опыты
сохраняют полные истории 200k пользователей. Основные ориентиры: manual+graph
≈0,505676 на исходном sampled test; после full-catalog calibration на отдельной
группе 512 пользователей — graph ≈0,145602 против manual ≈0,130728.
Первые 128 пользователей не подтвердили raw full-catalog выигрыш; этот результат
также должен сохраниться в выводах.

## Новая генерация вместо replay

```sh
python -m shared.download_proposer
python -m task1.promptfe_run --seed 42 --syntax infix --proposals 12 --iterations 350 --model .cache/models/qwen2.5-coder-1.5b-instruct --out artifacts/feature_search_new
```

Это новый proposal set. Даже с тем же seed CPU/CUDA и разные версии Torch могут
дать другие тексты; он не заменяет точное воспроизведение опубликованной таблицы.
Текущий вычислитель выражений находится в `task1/expression_engine.py` и не
загружает сторонний evaluator. Исторические random expressions для точного
повторения берутся из replay, поскольку fresh random generator теперь отдельный.

## Технические параметры и измеренные затраты

Идентификаторы случайных инициализаций основного сравнения:42/43/44; поиск выражений и его основной replay:42. В отчёте результаты нескольких запусков усредняются; точные значения seeds остаются здесь и в протоколах.

| Исходный этап | Зафиксированное время |
| --- | --- |
| Поиск окон, CPU, три запуска | 330 с |
| Поиск выражений с LLM, CPU | 1934 с |
| Поиск выражений, Kaggle | 2217 с |
| Replay сохранённых предложений | 665 с |

Это исторические измерения конкретных окружений; часть CPU-задач выполнялась параллельно. Таблица помогает оценить объём вычислений, но не является чистым benchmark ускорения между CPU/GPU. Подробные значения сохранены в runtime.json соответствующего этапа.
