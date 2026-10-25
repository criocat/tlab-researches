# Временная проверка взвешенной next-target метрики

Те же девять фиксированных политик оценены на исходной validation 9–15 августа. Модели не переобучаются, выбор не меняется. Это дополнительный период для диагностики; validation уже использовалась для отбора checkpoint, поэтому её нельзя считать новой независимой выборкой. Вторичная метрика сформулирована после исходного test.

Как и [на test](OBJECTIVE_DIAGNOSTIC.md), считаем `sum(w_i * ndcg_i) / sum(w_i)`, где `w_i` — фактическая доля просмотра записанного next-target. Взвешиваются пользователи; это не reward новой выдачи и не off-policy оценка.

## Изменение к своему reference на двух периодах

| policy | split | delta | CI low | CI high |
| --- | --- | --- | --- | --- |
| original_watch_rl | validation | -0.008018 | -0.018508 | 0.003120 |
| original_watch_rl | test | 0.008172 | -0.002084 | 0.018850 |
| original_balanced_rl | validation | 0.001443 | -0.003279 | 0.006568 |
| original_balanced_rl | test | 0.000881 | -0.003196 | 0.005031 |
| original_continued_sft | validation | 0.000259 | -0.003090 | 0.003723 |
| original_continued_sft | test | -0.003733 | -0.007309 | -0.000155 |
| full_monte_carlo | validation | 0.000773 | -0.003165 | 0.004391 |
| full_monte_carlo | test | 0.003963 | 0.000613 | 0.007664 |
| full_exact | validation | 0.000227 | -0.003502 | 0.003850 |
| full_exact | test | 0.004042 | 0.000417 | 0.008013 |

Все интервалы — paired user bootstrap с пересэмплированием weights и denominator. У MC/exact сначала усредняются пользовательские метрики трёх RL seed; SFT/RM фиксированы. Ни дополнительный период, ни вторичная метрика не отменяют исходный строгий validation fallback.

## Полные validation-результаты

| policy | group | users | ndcg@10 | recall@10 |
| --- | --- | --- | --- | --- |
| original_sft | all | 2000 | 0.110352 | 0.192500 |
| original_sft | watch=0 | 266 | 0.106930 | 0.191729 |
| original_sft | 0<watch<0.5 | 1026 | 0.088172 | 0.159844 |
| original_sft | watch>=0.5 | 708 | 0.143779 | 0.240113 |
| original_sft | observed_target_watch_weighted | 2000 | 0.128355 | 0.219299 |
| original_watch_rl | all | 2000 | 0.098960 | 0.194500 |
| original_watch_rl | watch=0 | 266 | 0.075373 | 0.180451 |
| original_watch_rl | 0<watch<0.5 | 1026 | 0.081435 | 0.155945 |
| original_watch_rl | watch>=0.5 | 708 | 0.133218 | 0.255650 |
| original_watch_rl | observed_target_watch_weighted | 2000 | 0.120337 | 0.232044 |
| original_balanced_rl | all | 2000 | 0.105725 | 0.190000 |
| original_balanced_rl | watch=0 | 266 | 0.087923 | 0.184211 |
| original_balanced_rl | 0<watch<0.5 | 1026 | 0.082689 | 0.153021 |
| original_balanced_rl | watch>=0.5 | 708 | 0.145796 | 0.245763 |
| original_balanced_rl | observed_target_watch_weighted | 2000 | 0.129798 | 0.224489 |
| original_continued_sft | all | 2000 | 0.110258 | 0.194000 |
| original_continued_sft | watch=0 | 266 | 0.105825 | 0.187970 |
| original_continued_sft | 0<watch<0.5 | 1026 | 0.087697 | 0.164717 |
| original_continued_sft | watch>=0.5 | 708 | 0.144617 | 0.238701 |
| original_continued_sft | observed_target_watch_weighted | 2000 | 0.128614 | 0.217676 |
| masked_sft | all | 2000 | 0.126386 | 0.213000 |
| masked_sft | watch=0 | 266 | 0.118090 | 0.206767 |
| masked_sft | 0<watch<0.5 | 1026 | 0.105334 | 0.187135 |
| masked_sft | watch>=0.5 | 708 | 0.160010 | 0.252825 |
| masked_sft | observed_target_watch_weighted | 2000 | 0.145585 | 0.237182 |
| capped_graph_masked | all | 2000 | 0.129941 | 0.219000 |
| capped_graph_masked | watch=0 | 266 | 0.121232 | 0.206767 |
| capped_graph_masked | 0<watch<0.5 | 1026 | 0.105015 | 0.190058 |
| capped_graph_masked | watch>=0.5 | 708 | 0.169335 | 0.265537 |
| capped_graph_masked | observed_target_watch_weighted | 2000 | 0.152580 | 0.242602 |
| full_graph_masked | all | 2000 | 0.135639 | 0.223500 |
| full_graph_masked | watch=0 | 266 | 0.125799 | 0.236842 |
| full_graph_masked | 0<watch<0.5 | 1026 | 0.109457 | 0.190058 |
| full_graph_masked | watch>=0.5 | 708 | 0.177277 | 0.266949 |
| full_graph_masked | observed_target_watch_weighted | 2000 | 0.159070 | 0.245606 |
| full_monte_carlo | all | 2000 | 0.133102 | 0.220667 |
| full_monte_carlo | watch=0 | 266 | 0.111256 | 0.196742 |
| full_monte_carlo | 0<watch<0.5 | 1026 | 0.107719 | 0.187784 |
| full_monte_carlo | watch>=0.5 | 708 | 0.178093 | 0.277307 |
| full_monte_carlo | observed_target_watch_weighted | 2000 | 0.159842 | 0.254520 |
| full_exact | all | 2000 | 0.132548 | 0.219500 |
| full_exact | watch=0 | 266 | 0.109508 | 0.192982 |
| full_exact | 0<watch<0.5 | 1026 | 0.107513 | 0.186485 |
| full_exact | watch>=0.5 | 708 | 0.177484 | 0.277307 |
| full_exact | observed_target_watch_weighted | 2000 | 0.159296 | 0.253842 |

Модели заново предсказали метрики исходной validation: сохранённые раньше результаты контрольных SFT/RL вариантов совпали. Все checkpoint SHA и новые per-user таблицы приложены. Воспроизведение: `python -m task2.objective_validation` после исходных этапов и `task2.relevance_objectives`.
