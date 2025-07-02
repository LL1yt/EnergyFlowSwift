#!/usr/bin/env python3
"""
Пример миграции модуля с hardcoded значениями на использование конфига
=====================================================================

Показывает как переписать код с hardcoded значениями.
"""

import torch
import torch.nn as nn
from new_rebuild.config import get_project_config
from new_rebuild.utils import strict_no_hardcoded, no_hardcoded, HardcodedValueError


# ❌ ПЛОХО: Старый код с hardcoded значениями
class OldModelWithHardcoded(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Hardcoded значения везде!
        hidden_dim = 64  # ❌
        dropout_rate = 0.1  # ❌
        num_layers = 3  # ❌
        
        self.layers = nn.ModuleList([
            nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = torch.relu(x)
            x = self.dropout(x)
        return x


# ✅ ХОРОШО: Новый код с использованием конфига
class NewModelWithConfig(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        config = get_project_config()
        
        # Все параметры из конфига!
        hidden_dim = config.model.hidden_dim
        dropout_rate = config.architecture.cnf_dropout_rate
        num_layers = config.model.num_layers
        
        self.layers = nn.ModuleList([
            nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = torch.relu(x)
            x = self.dropout(x)
        return x


# ❌ ПЛОХО: Функция обучения с hardcoded
def train_model_old(model, data_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)  # ❌
    max_epochs = 10  # ❌
    log_every = 100  # ❌
    
    for epoch in range(max_epochs):
        for batch_idx, (data, target) in enumerate(data_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = nn.functional.mse_loss(output, target)
            loss.backward()
            
            # Gradient clipping с hardcoded значением
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # ❌
            
            optimizer.step()
            
            if batch_idx % log_every == 0:  # ❌
                print(f"Loss: {loss.item()}")


# ✅ ХОРОШО: Функция обучения с конфигом
@no_hardcoded  # Декоратор будет проверять аргументы
def train_model_new(model, data_loader):
    config = get_project_config()
    
    # Все параметры из конфига
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.training_optimizer.learning_rate,
        weight_decay=config.training_optimizer.weight_decay
    )
    
    max_epochs = config.training.num_epochs
    log_every = config.training_optimizer.log_batch_frequency
    
    for epoch in range(max_epochs):
        for batch_idx, (data, target) in enumerate(data_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = nn.functional.mse_loss(output, target)
            loss.backward()
            
            # Gradient clipping из конфига
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                max_norm=config.training_optimizer.gradient_clip_max_norm
            )
            
            optimizer.step()
            
            if batch_idx % log_every == 0:
                print(f"Loss: {loss.item()}")


# Пример с использованием strict_no_hardcoded для постепенной миграции
class TransitionModel(nn.Module):
    """Модель в процессе миграции - использует strict_no_hardcoded"""
    
    def __init__(self, input_dim):
        super().__init__()
        
        # strict_no_hardcoded автоматически заменит на значение из конфига
        hidden_dim = strict_no_hardcoded(64, "model.hidden_dim")
        dropout_rate = strict_no_hardcoded(0.1, "architecture.cnf_dropout_rate")
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Функция для демонстрации обработки ошибок
def process_data_with_checks(data, threshold=0.8):  # 0.8 это hardcoded!
    """Пример функции которая проверяет hardcoded значения"""
    
    try:
        # Это вызовет ошибку
        check_hardcoded_value(threshold, "process_data threshold")
    except HardcodedValueError as e:
        print(f"Обнаружено hardcoded значение! {e}")
        # Получаем из конфига
        config = get_project_config()
        threshold = config.embedding_mapping.surface_coverage
        
    # Обработка данных с правильным threshold
    return data[data > threshold]


def demonstrate_migration():
    """Демонстрация процесса миграции"""
    print("🔄 Демонстрация миграции с hardcoded на config")
    print("=" * 60)
    
    from new_rebuild.config import create_experiment_config, set_project_config
    config = create_experiment_config()
    set_project_config(config)
    
    print("\n1. Старая модель с hardcoded (плохо):")
    old_model = OldModelWithHardcoded(100)
    print(f"   ❌ Использует hardcoded hidden_dim=64, dropout=0.1")
    
    print("\n2. Новая модель с конфигом (хорошо):")
    new_model = NewModelWithConfig(100)
    print(f"   ✅ Использует config: hidden_dim={config.model.hidden_dim}, "
          f"dropout={config.architecture.cnf_dropout_rate}")
    
    print("\n3. Модель в процессе миграции (с strict_no_hardcoded):")
    transition_model = TransitionModel(100)
    print(f"   🔄 Автоматически заменяет hardcoded на значения из конфига")
    
    print("\n4. Обработка обнаруженных hardcoded значений:")
    import torch
    data = torch.rand(10)
    result = process_data_with_checks(data)
    print(f"   ✅ Threshold взят из конфига: {config.embedding_mapping.surface_coverage}")
    
    print("\n" + "=" * 60)
    print("✅ Миграция продемонстрирована!")


if __name__ == "__main__":
    demonstrate_migration()