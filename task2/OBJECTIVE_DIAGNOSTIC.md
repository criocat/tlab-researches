# Разные цели: следующий объект и доля просмотра

Это дополнительная диагностика на уже изученном test. Модели, параметры и validation-выбор сохранены. Публикуются все девять заранее перечисленных политик; веса не подбираются.

Обычная next-target оценка одинаково учитывает любое записанное взаимодействие. В KION 260 из 2000 test-targets имеют нулевую долю просмотра, 1003 — от нуля до 0,5, ещё 737 — не менее 0,5. Поэтому максимизация completion-proxy и извлечение любого следующего target задают разные предпочтения.

Для анализа считаем **среднее per-user target NDCG, взвешенное фактической долей просмотра target**: `sum(w_i * ndcg_i) / sum(w_i)`. Аналогично считаем weighted Recall. Это веса между пользователями, а не переопределённый graded NDCG внутри запроса. Они используют наблюдаемый outcome записанной цели; это не reward нового рекомендованного списка и не off-policy оценка ценности.

| policy | ordinary NDCG | watch-weighted mean NDCG | ordinary Recall | watch-weighted Recall |
| --- | --- | --- | --- | --- |
| original_sft | 0.095374 | 0.110104 | 0.164500 | 0.189664 |
| original_watch_rl | 0.093601 | 0.118277 | 0.164500 | 0.198554 |
| original_balanced_rl | 0.092107 | 0.110986 | 0.157500 | 0.187536 |
| original_continued_sft | 0.092981 | 0.106371 | 0.159500 | 0.183243 |
| masked_sft | 0.107339 | 0.124655 | 0.176500 | 0.202720 |
| capped_graph_masked | 0.113550 | 0.136967 | 0.185500 | 0.219310 |
| full_graph_masked | 0.116802 | 0.137697 | 0.194000 | 0.221830 |
| full_monte_carlo | 0.116768 | 0.141660 | 0.192500 | 0.225008 |
| full_exact | 0.116037 | 0.141740 | 0.190667 | 0.224890 |

## Парные изменения к соответствующему reference

| policy | baseline | metric | delta | CI low | CI high |
| --- | --- | --- | --- | --- | --- |
| original_watch_rl | original_sft | unweighted | -0.001773 | -0.008858 | 0.005721 |
| original_watch_rl | original_sft | target_watch_weighted | 0.008172 | -0.002084 | 0.018850 |
| original_balanced_rl | original_sft | unweighted | -0.003267 | -0.006706 | 0.000200 |
| original_balanced_rl | original_sft | target_watch_weighted | 0.000881 | -0.003196 | 0.005031 |
| original_continued_sft | original_sft | unweighted | -0.002393 | -0.004810 | 0.000062 |
| original_continued_sft | original_sft | target_watch_weighted | -0.003733 | -0.007309 | -0.000155 |
| full_monte_carlo | full_graph_masked | unweighted | -0.000034 | -0.002487 | 0.002737 |
| full_monte_carlo | full_graph_masked | target_watch_weighted | 0.003963 | 0.000613 | 0.007664 |
| full_exact | full_graph_masked | unweighted | -0.000766 | -0.003440 | 0.002235 |
| full_exact | full_graph_masked | target_watch_weighted | 0.004042 | 0.000417 | 0.008013 |

MC/exact на full graph reference имеют положительный watch-weighted point estimate и условные user-bootstrap CI. Это согласуется с несовпадением обучаемых целей, но не доказывает причинного улучшения рекомендаций. Веса и вторичная метрика исследованы после исходного test; множественные проверки и фиксированные SFT/RM ограничивают CI. Никакой новый test-победитель не назначается: строгий validation-отбор сохраняет SFT.

## Группы наблюдаемой доли просмотра

| policy | group | users | ndcg@10 | recall@10 |
| --- | --- | --- | --- | --- |
| original_sft | watch=0 | 260 | 0.096644 | 0.153846 |
| original_sft | 0<watch<0.5 | 1003 | 0.077820 | 0.139581 |
| original_sft | watch>=0.5 | 737 | 0.118816 | 0.202171 |
| original_watch_rl | watch=0 | 260 | 0.076500 | 0.153846 |
| original_watch_rl | 0<watch<0.5 | 1003 | 0.074783 | 0.130608 |
| original_watch_rl | watch>=0.5 | 737 | 0.125244 | 0.214383 |
| original_balanced_rl | watch=0 | 260 | 0.082849 | 0.134615 |
| original_balanced_rl | 0<watch<0.5 | 1003 | 0.073732 | 0.129611 |
| original_balanced_rl | watch>=0.5 | 737 | 0.120380 | 0.203528 |
| original_continued_sft | watch=0 | 260 | 0.097169 | 0.153846 |
| original_continued_sft | 0<watch<0.5 | 1003 | 0.075872 | 0.133599 |
| original_continued_sft | watch>=0.5 | 737 | 0.114788 | 0.196744 |
| masked_sft | watch=0 | 260 | 0.103919 | 0.165385 |
| masked_sft | 0<watch<0.5 | 1003 | 0.088298 | 0.151545 |
| masked_sft | watch>=0.5 | 737 | 0.134460 | 0.214383 |
| capped_graph_masked | watch=0 | 260 | 0.103505 | 0.176923 |
| capped_graph_masked | 0<watch<0.5 | 1003 | 0.090838 | 0.151545 |
| capped_graph_masked | watch>=0.5 | 737 | 0.148002 | 0.234735 |
| full_graph_masked | watch=0 | 260 | 0.112549 | 0.196154 |
| full_graph_masked | 0<watch<0.5 | 1003 | 0.094500 | 0.164506 |
| full_graph_masked | watch>=0.5 | 737 | 0.148655 | 0.233379 |
| full_monte_carlo | watch=0 | 260 | 0.101321 | 0.178205 |
| full_monte_carlo | 0<watch<0.5 | 1003 | 0.094623 | 0.162845 |
| full_monte_carlo | watch>=0.5 | 737 | 0.152356 | 0.237901 |
| full_exact | watch=0 | 260 | 0.099498 | 0.175641 |
| full_exact | 0<watch<0.5 | 1003 | 0.093500 | 0.159189 |
| full_exact | watch>=0.5 | 737 | 0.152542 | 0.238806 |

Группы определены последующим наблюдаемым outcome, поэтому различия между ними не являются причинной оценкой эффекта. При трёх RL seed сначала усредняются пользовательские метрики, затем пересэмплируются пользователи вместе с их весами. Вариативность обучения SFT/RM этим интервалом не охватывается.

Воспроизведение: `python -m task2.relevance_objectives`. Нужны сохранённые per-user CSV и исходный `artifacts/task2/examples_test.npz`; для чтения и проверки взвешенных сумм достаточно небольших таблиц из архива.

[Проверка на validation](OBJECTIVE_VALIDATION.md) дала более слабый weighted прирост MC/exact с CI через ноль. Эффект на втором периоде не подтверждён убедительно; исходная validation также уже использовалась для выбора checkpoint.
