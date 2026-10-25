# Воспроизведение задания 2

[Установка и данные](ENVIRONMENT.md) · [Отчёт](../task2/report.pdf).

[Сохранённые локальные результаты](../result_artifacts/task2/metrics.csv) · [Протокол](../result_artifacts/task2/protocol.json).

## Один основной запуск с seed 42

```sh
python reproduce.py task2 --dry-run
python reproduce.py task2
```

Скрипт обучает компактную GPT-2 на 100k переходов, две reward models и пять
заранее заданных RL-вариантов. Затем выполняет matched continued-SFT контроль
и сохраняет таблицы. Все параметры исходного основного опыта оставлены прежними:
50k пользователей, контекст 32 IDs, 8 SFT epochs, 5 RM epochs, 200 RL updates,
2000 validation/test пользователей, основной seed 42. RL sampling/minibatches
используют фиксированное смещение seed на 100; auditor имеет отдельную
зафиксированную инициализацию, указанную в коде и протоколах.

Выходы — `artifacts/reproduced/task2/` и `artifacts/reproduced/task2_results.md`.
Ориентиры исходного CPU опыта: SFT NDCG@10 ≈0,095374, выбранный balanced RL
≈0,092107. Это исходный опыт; более сильный reference и остановка на 50 шагах
относятся к дополнительным этапам ниже.

Словарь фиксируется до 9 августа; train заканчивается 8 августа,
validation — 9–15 августа, test — 16–22 августа. Внутри дня порядок неизвестен,
поэтому target видит только предыдущие дни. Cold targets учитываются как misses.
Рекомендации оцениваются по полному train-каталогу, без подмешивания target.

## Полное повторение дополнений

Дополнения используют веса и матрицы предыдущих этапов.
Все они создаются приведёнными ниже командами в свежем клоне.

Выполнить по порядку:

```sh
python -m shared.download
python -m task2.run --users 50000 --max-train 100000 --max-eval 2000 --epochs 8 --reward-epochs 5 --rl-steps 200 --seed 42
python -m task2.control
python -m shared.analyze --task task2
python -m task2.factorial
python -m task2.decoding_study
python -m task2.robust_reward
python -m task2.analytic_policy
python -m task2.history_mask
python -m task2.graph_warmstart
python -m task2.context_reward
python -m task2.context_rl
python -m task2.context_rl --warmstart-selected --out artifacts/task2_joint
python -m task2.reward_ablation
python -m task2.conditional_bandit
python -m task2.return_reward
python -m task2.full_data_control
python -m task2.conditional_bandit --full-reference --out artifacts/task2_conditional_full_reference
python -m task2.relevance_objectives
python -m task2.objective_validation
python -m task2.validation_path
```

Полный train содержит 181 733 перехода; исходные validation/test arrays при
расширении train сохраняются точно. Ориентиры CPU: full graph+mask SFT
NDCG≈0,116802; выбранный по validation exact RL на 50 updates — средний
NDCG≈0,117206, watch-proxy≈0,434011 против reference≈0,425677.
CI разницы NDCG к reference и matched continued-SFT пересекают ноль.
Более высокий predicted reward не доказывает реальный отклик или удержание.

## Отдельная GPU-репликация SFT

Для GPU-строк отчёта использовались Tesla T4, Torch 2.11.0+cu128,
NumPy 2.1.3, Pandas 2.3.3, SciPy 1.16.3 и scikit-learn 1.6.1.
На Kaggle запускать из копии этого репозитория с CUDA Torch:

```sh
unset CUDA_VISIBLE_DEVICES
python -m shared.download
python -m task2.sft_multiseed --out artifacts/kaggle_sft_multiseed/tlab/artifacts/task2_sft_multiseed
python -m task2.sft_multiseed --full-train --out artifacts/kaggle_sft_full_multiseed/tlab/artifacts/task2_sft_multiseed
python -m shared.sft_replication_analysis
```

Это два дополнительных опыта по 9 fits, а не условие основного запуска.
Скрипт сверяет SHA массивов с `task2/evaluation_manifest.json` до обучения.
CPU и GPU результаты приводятся отдельно; seed не гарантирует одинаковых весов
между разными Torch/device. Для обычного воспроизведения достаточно одного
основного CPU запуска с seed 42; результаты GPU приведены в PDF.

## Технические параметры и измеренные затраты

Основной seed:42; повторные RL-запуски:42/43/44. Это идентификаторы генераторов случайных чисел; в основном отчёте вместо них указано число повторных обучений и смысл интервалов.

| Исходный этап | Зафиксированное время |
| --- | --- |
| Основной CPU pipeline | 506.2 с |
| Дополнительные RL-комбинации | 344.0 с |
| Декодирование | 14.5 с |
| Условный reward gradient | 22.2 с |
| Выбор момента остановки | 76.5 с |

Значения относятся к записанным CPU-окружениям и условиям кеша; часть задач делила ресурсы. GPU-версии, параметры и отдельные времена хранятся в protocol/runtime JSON. Эти числа не являются гарантией времени повторения на другой машине.
