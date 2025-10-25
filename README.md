# LLM Training Sandbox - DDP Demo

Минимальный пример обучения LLM модели с Distributed Data Parallel (DDP) для демонстрации студентам.

## Особенности

- Простая трансформер модель
- Tiny Stories датасет с TikTok tokenizer
- DDP обертка с torch.distributed
- Torch profiler и memory profiler
- Запуск через torchrun
- Минимум кода, максимум понимания

## Установка

```bash
pip install -r requirements.txt
```

## Запуск

### Одно GPU
```bash
python train.py --batch_size 4 --num_epochs 1
```

### Множественные GPU (DDP)
```bash
./run_training.sh 2  # 2 GPU
./run_training.sh 4  # 4 GPU
```

### С профилированием
```bash
./run_training.sh 2  # Включает --profile и --memory_profile
```

## Структура проекта

- `model.py` - Простая трансформер модель
- `dataset.py` - Загрузка Tiny Stories с TikTok tokenizer
- `train.py` - Основной скрипт обучения с DDP
- `run_training.sh` - Скрипт запуска через torchrun
- `requirements.txt` - Зависимости

## Профилирование

После запуска с `--profile`:
- Результаты torch.profiler: `./profiler_logs/`
- Просмотр: `tensorboard --logdir=./profiler_logs`

После запуска с `--memory_profile`:
- Снимки памяти: `memory_snapshot_*.pickle`
- Анализ: `torch.cuda.memory._dump_snapshot()`
