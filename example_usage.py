#!/usr/bin/env python3
"""
Пример использования LLM Training Sandbox

Этот скрипт демонстрирует основные компоненты проекта
"""

import torch
from model import SimpleTransformer
from dataset import TinyStoriesDataset

def demo_model():
    """Демонстрация модели"""
    print("=== Демонстрация модели ===")
    
    # Создаем модель
    model = SimpleTransformer(
        vocab_size=100256,
        d_model=512,
        nhead=8,
        num_layers=6,
        max_len=512
    )
    
    print(f"Модель создана: {sum(p.numel() for p in model.parameters()):,} параметров")
    
    # Тестовый forward pass
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
    
    print(f"Вход: {input_ids.shape} -> Выход: {logits.shape}")
    print("✓ Модель работает корректно\n")


def demo_dataset():
    """Демонстрация датасета"""
    print("=== Демонстрация датасета ===")
    
    try:
        # Создаем датасет
        dataset = TinyStoriesDataset(max_length=128)
        print(f"Датасет загружен: {len(dataset)} примеров")
        
        # Получаем один пример
        sample = dataset[0]
        print(f"Пример:")
        print(f"  input_ids: {sample['input_ids'].shape}")
        print(f"  attention_mask: {sample['attention_mask'].shape}")
        print(f"  labels: {sample['labels'].shape}")
        print("✓ Датасет работает корректно\n")
        
    except Exception as e:
        print(f"⚠️  Датасет не загружен (возможно, нет интернета): {e}\n")


def demo_training_command():
    """Демонстрация команд запуска"""
    print("=== Команды запуска ===")
    
    commands = [
        "# Установка зависимостей",
        "pip install -r requirements.txt",
        "",
        "# Одно GPU",
        "python train.py --batch_size 4 --num_epochs 1",
        "",
        "# DDP на 2 GPU",
        "./run_training.sh 2",
        "",
        "# DDP на 4 GPU с профилированием",
        "./run_training.sh 4",
        "",
        "# Просмотр профилирования",
        "tensorboard --logdir=./profiler_logs"
    ]
    
    for cmd in commands:
        print(cmd)
    
    print("\n✓ Все команды готовы к использованию\n")


if __name__ == "__main__":
    print("🚀 LLM Training Sandbox - Демонстрация\n")
    
    demo_model()
    demo_dataset()
    demo_training_command()
    
    print("🎉 Демонстрация завершена!")
    print("Для реального обучения запустите: ./run_training.sh 2")
