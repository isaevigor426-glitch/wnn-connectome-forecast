# Запуск

Рабочий корень — `alfa_codex_handoff`. Reference не изменён.
Основной результат: `notebooks/LSTM_Colab_T4.ipynb`. Ноутбук сгенерирован
`scripts/make_notebook.py`; при изменении генератора пересоздайте ноутбук.

## Google Colab T4 — вручную

1. Скопируйте пакет на Drive в `MyDrive/alfa_codex_handoff`, исключив `.venv`,
   `__pycache__`, `.pytest_cache`. Скопируйте предоставленные `.npy` из
   `for_codex/valid300` в `MyDrive/alfa_codex_handoff/assets/valid300`.
2. Откройте `notebooks/LSTM_Colab_T4.ipynb` в Colab, выберите GPU T4.
3. В первой ячейке настройте PROJECT, RAW, DATA, ARCHIVE, SEARCH, OUT.
   DATA и ARCHIVE по умолчанию на Drive. Подготовленные массивы занимают
   около 58 GB; архив — 33.74 GB, плюс место для исходных Parquet и checkpoints.
   Проверьте реальную квоту Drive через интерфейс: filesystem free может быть неточным.
4. Выполните ячейки сверху вниз. Загрузка запускается только вами в Colab.
   Уже существующие исправные Parquet берутся из SEARCH. Train не требует is_scored.
   Повреждённый valid исключается. HTTP/1.1, retries, Range resume; полный gzip CRC
   и tar scan выполняются до извлечения. ETag сервера не является SHA256 файла.
5. После отключения подключите Drive и повторите ячейки. DATA/manifest.json
   позволяет пропустить скачивание и подготовку; OUT/last.json включает resume.
   Если подготовка была прервана, manifest отсутствует: удалите только неполные
   prepared `*_X/Y/M.npy`, либо выберите новый DATA (для повторной подготовки нужен диск).
6. Обучение 5 эпох: LSTM 2×128, batch 32 (последний неполный включён), chunk 1000,
   Adam, OneCycleLR 2e-3 / pct_start=.05, clip_grad=1, seed=0, float32 без AMP.
   Train=10607 + 1573 valid; 300 sorted holdout не участвуют. Loss использует t>=99,
   не is_scored. Прогресс по batch с loss и приблизительным ETA. Валидация каждую эпоху.
7. `last.json` / `best.json` указывают на проверенные SHA256 версионные `.pt`.
   Запись после полного batch, по умолчанию каждого; при обрыве незакоммиченный batch
   повторяется целиком. Сохраняются Adam, scheduler, RNG, shuffle, следующая позиция.
   Две версии остаются на Drive; при повреждённом checksum загрузка останавливается,
   автоматической загрузки сомнительных весов нет. Загружайте только собственные `.pt`.
8. Финальные ячейки вычисляют best predictions, честное сравнение, экспорт LSTM,
   проверку на реальных X valid300 и короткий CPU benchmark. Эти действия после
   обучения пользователь запускает вручную. Полное исполнение на T4 пока не проверено.

## CLI

```bash
python scripts/acquire_data.py --archive /content/drive/MyDrive/alfa_data/starter.tar.gz --out /content/data/raw --search /content/wnn_connectome_starterpack /content/drive/MyDrive/alfa_data
python scripts/prepare_data.py --source /content/data/raw --out /content/drive/MyDrive/alfa_data/prepared
python src/train.py --model lstm --hidden 128 --seed 0 --epochs 5 --data_dir /content/drive/MyDrive/alfa_data/prepared --out /content/drive/MyDrive/alfa_lstm_run
# После обрыва — те же параметры и --resume.
```

Для воспроизведения доступны `--model gru`, `gru_ev`, `gru_wide`.
Для wide явно задайте `--hidden 256`. Event features совпадают с `_evfeat`,
нормируются по train indices 0,100,...,1900. Семейство цели используется только
в offline метрике, не в inference. Порядок признаков сохраняется из Parquet.

## Проверка пользы LSTM

```bash
python scripts/validate.py --assets assets/valid300 --lstm RUN/P_lstm_valid300.npy --provenance RUN/prediction_provenance.json --out RUN/validation.json
python scripts/export_checkpoint.py --checkpoint-dir RUN --out RUN/lstm_weights.npz
python scripts/verify_export.py --weights RUN/lstm_weights.npz --data-dir DATA --prediction RUN/P_lstm_valid300.npy --out RUN/verification.json
python scripts/benchmark.py --weights RUN/lstm_weights.npz --out RUN/benchmark.json
```

Замените RUN/DATA на реальные пути. Проверяются sorted holdout и SHA256 Y/M;
raw LSTM predictions сохраняются float32, clipping применяется только в метрике.
`validate.py` показывает score по целям, прирост равновесного ансамбля и парный
стратифицированный sequence bootstrap (1000 повторов). Прирост >=.0015 — минимальный
порог; <.003 остаётся неустойчивым, независимая проверка желательна.

## Точный ансамбль и ZIP

Нельзя принять качество ансамбля как качество одиночной LSTM. Нужны checkpoints
всех выбранных независимых моделей; отдельные checkpoints honest3 отсутствуют.
Для собственных моделей с общим holdout:

```bash
python scripts/export_ensemble.py --checkpoints RUN1 RUN2 --out candidate.npz
python scripts/predict_ensemble.py --checkpoints RUN1 RUN2 --data-dir DATA --out candidate_predictions.npy
python scripts/validate.py --assets assets/valid300 --lstm candidate_predictions.npy --provenance candidate_predictions.npy.json --candidate --out candidate_quality.json
python scripts/verify_export.py --weights candidate.npz --data-dir DATA --prediction candidate_predictions.npy --out candidate_verify.json
python scripts/benchmark.py --weights candidate.npz --out candidate_benchmark.json
python scripts/build_submission.py --weights candidate.npz --validation candidate_quality.json --verification candidate_verify.json --benchmark candidate_benchmark.json --out submission.zip
```

Равные коэффициенты задаются явно экспортёром; cross_mix=0 для новых моделей.
Старые архивы без cross_mix сохраняют обёртку .07. Движок поддерживает старые typ 0/1/2;
экспортёр собственных checkpoints — 0/2/3. Partner не включается в честную оценку.
ZIP содержит только solution.py и weights.npz. Сборка требует совпадения hash весов,
прогнозов, качества точного кандидата, полной проверки 300 последовательностей и
benchmark на одном закреплённом CPU core: ratio <=1.4, прогноз <=54 минут на 39.4M строк.
Это консервативный допуск, не гарантия официального раннера.

## Локальные короткие тесты

```bash
python -m venv .venv
.venv/bin/pip install numpy torch numba pyarrow tqdm pytest matplotlib
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest tests -q
```

Фактическое окружение локальной проверки записано в `tests/environment.txt`.
Локальное обучение на данных блокируется проверкой T4. Синтетические тесты обучения
короткие. `tests/local_benchmark.json` — Mac ARM, не целевой раннер.
Правило private leaderboard не подтверждено; нужен официальный ответ организаторов.
Mamba/независимые partner predictions отсутствуют. Финального submission.zip нет.
