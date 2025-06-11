"""
Dynamic Configuration System для 3D Cellular Neural Network
Основано на биологических данных вентролатеральной префронтальной коры (vlPFC)
"""

import math
import yaml
import torch
from dataclasses import dataclass
from typing import Dict, Any, Optional, Union
from pathlib import Path
import logging
import re

logger = logging.getLogger(__name__)


@dataclass
class BiologicalConstants:
    """Биологические константы vlPFC"""

    # Нейроанатомические данные vlPFC
    neurons_one_hemisphere: int = 93_750_000
    neurons_both_hemispheres: int = 180_000_000
    target_neurons_average: int = 136_875_000

    # Синаптические характеристики
    synapses_per_neuron_min: int = 5_000
    synapses_per_neuron_max: int = 15_000
    synapses_per_neuron_avg: int = 10_000

    # Структурные пропорции
    depth_to_width_ratio: float = 0.5  # depth = 0.5 × width

    # Базовые размеры решетки для 100% масштаба
    base_width: int = 666
    base_height: int = 666

    @property
    def base_depth(self) -> int:
        """Биологически корректная глубина"""
        return int(self.base_width * self.depth_to_width_ratio)


@dataclass
class ScaleSettings:
    """Настройки масштабирования для разных режимов"""

    development: float = 0.01  # 1% - быстрая разработка
    research: float = 0.1  # 10% - исследования
    validation: float = 0.3  # 30% - валидация
    production: float = 1.0  # 100% - продакшен

    def get_scale(self, mode: str) -> float:
        """Получить коэффициент масштабирования для режима"""
        return getattr(self, mode, self.development)


class ExpressionEvaluator:
    """Вычислитель выражений для динамической конфигурации"""

    def __init__(self):
        self.bio_constants = BiologicalConstants()

    def smart_round(self, value: float) -> int:
        """Умное округление для получения целых чисел"""
        return int(round(value))

    def evaluate_expression(self, expr: str, context: Dict[str, Any]) -> Any:
        """Вычислить выражение в контексте переменных"""
        if (
            not isinstance(expr, str)
            or not expr.startswith("{")
            or not expr.endswith("}")
        ):
            return expr

        # Убираем фигурные скобки
        expression = expr[1:-1]

        # Добавляем функции в контекст
        eval_context = {
            **context,
            "smart_round": self.smart_round,
            "round": round,
            "int": int,
            "float": float,
            "min": min,
            "max": max,
            "math": math,
        }

        try:
            result = eval(expression, {"__builtins__": {}}, eval_context)
            logger.debug(f"📐 Evaluated '{expr}' = {result}")
            return result
        except Exception as e:
            logger.error(f"[ERROR] Error evaluating expression '{expr}': {e}")
            return expr

    def process_config_dict(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Рекурсивно обработать конфигурацию, вычислив все выражения"""

        def flatten_dict(d, parent_key="", sep="_"):
            """Превратить вложенный словарь в плоский для создания глобального контекста"""
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        # Создаем глобальный контекст со всеми простыми значениями
        global_context = {}

        def collect_simple_values(data, prefix=""):
            """Собрать все простые значения для контекста"""
            for key, value in data.items():
                full_key = f"{prefix}_{key}" if prefix else key

                if isinstance(value, dict):
                    collect_simple_values(value, full_key)
                elif isinstance(value, (int, float)) or (
                    isinstance(value, str) and not value.startswith("{")
                ):
                    global_context[key] = value  # Локальное имя
                    global_context[full_key] = value  # Полное имя

        # Собираем простые значения
        collect_simple_values(config)

        def process_section(data):
            """Обработать секцию конфигурации"""
            result = {}

            # Копируем простые значения и обрабатываем вложенные секции
            for key, value in data.items():
                if isinstance(value, dict):
                    result[key] = process_section(value)
                elif isinstance(value, (int, float)) or (
                    isinstance(value, str) and not value.startswith("{")
                ):
                    result[key] = value
                    global_context[key] = value  # Обновляем контекст
                else:
                    result[key] = value

            # Вычисляем выражения в несколько итераций
            max_iterations = 15
            for iteration in range(max_iterations):
                changed = False

                for key, value in data.items():
                    if isinstance(value, str) and value.startswith("{"):
                        new_value = self.evaluate_expression(value, global_context)
                        if new_value != result.get(key) and not isinstance(
                            new_value, str
                        ):
                            result[key] = new_value
                            global_context[key] = new_value  # Добавляем в контекст
                            changed = True

                if not changed:
                    break

            return result

        return process_section(config)


class DynamicConfigGenerator:
    """Генератор динамической конфигурации"""

    def __init__(self):
        self.bio_constants = BiologicalConstants()
        self.scale_settings = ScaleSettings()
        self.evaluator = ExpressionEvaluator()

    def detect_hardware_mode(self) -> str:
        """Автоматическое определение оптимального режима на основе GPU памяти"""
        try:
            if torch.cuda.is_available():
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (
                    1024**3
                )
                logger.info(f"[PC] Detected GPU memory: {gpu_memory_gb:.1f}GB")

                if gpu_memory_gb >= 20:
                    return "validation"  # RTX 5090+
                elif gpu_memory_gb >= 12:
                    return "research"  # RTX 4070 Ti+
                else:
                    return "development"  # Меньше 12GB
            else:
                logger.warning("[WARNING] CUDA not available, using development mode")
                return "development"
        except Exception as e:
            logger.warning(
                f"[WARNING] Hardware detection failed: {e}, using development mode"
            )
            return "development"

    def create_base_config_template(self) -> Dict[str, Any]:
        """Создать базовый шаблон конфигурации с выражениями"""
        return {
            "lattice": {
                # Базовые размеры
                "x": self.bio_constants.base_width,
                "y": self.bio_constants.base_height,
                "z": "{smart_round(x*0.5)}",  # Биологически точная глубина
                # Масштабирование (будет заполнено позже)
                "scale_factor": 0.1,  # Placeholder
                "xs": "{smart_round(x*scale_factor)}",
                "ys": "{smart_round(y*scale_factor)}",
                "zs": "{smart_round(z*scale_factor)}",
                # Автоматические вычисления
                "total_neurons": "{xs * ys * zs}",
                "surface_size": "{xs * ys}",
                "volume": "{xs * ys * zs}",
            },
            "embeddings": {
                "embedding_dim": "{smart_round(xs*ys)}",  # = surface_size
                "teacher_embedding_dim": 768,
            },
            "training": {
                "batch_size": 1024,  # Будет скорректировано по режиму
                "learning_rate": 0.001,
                "epochs": 100,
            },
            "gmlp": {
                # Биологически правильное масштабирование
                # При scale=1.0 → ~10,000 параметров, при scale=0.06 → ~600 параметров
                "target_params": "{smart_round(10000 * scale_factor)}",  # Биологическое количество синапсов
                "neighbor_count": 6,  # Всегда 6 для 3D решетки
                # Биологически правильные размеры на основе scale_factor:
                "state_size": "{smart_round(max(8, min(32, target_params ** 0.5 / 3)))}",  # Масштабируемый state
                "hidden_dim": "{smart_round(max(8, min(128, target_params ** 0.5 / 4)))}",  # Масштабируемый hidden
                "external_input_size": "{smart_round(max(4, min(12, target_params ** 0.5 / 8)))}",  # Масштабируемый input
                "memory_dim": "{smart_round(max(4, min(32, target_params ** 0.5 / 6)))}",  # Масштабируемая память
                # Фиксированные параметры
                "use_memory": True,
                "activation": "gelu",
                "dropout": 0.1,
            },
            # НОВАЯ СЕКЦИЯ: Minimal NCA Configuration
            "nca": {
                # NCA параметрическая эффективность: target 68-10000 параметров
                "enabled": True,  # Флаг для переключения gMLP → NCA
                "target_params": "{smart_round(max(68, min(10000, 10000 * scale_factor)))}",  # Масштабируется как gMLP
                "neighbor_count": 6,  # Совместимо с 3D решеткой
                # Масштабируемые размеры для NCA (формулы оптимизированы для NCA архитектуры):
                "state_size": "{smart_round(max(4, min(20, (target_params / 8) ** 0.6)))}",  # Оптимизированное масштабирование
                "hidden_dim": "{smart_round(max(2, min(12, (target_params / 12) ** 0.5)))}",  # Меньше чем gMLP
                "external_input_size": "{smart_round(max(1, min(6, target_params / 50)))}",  # Линейное масштабирование
                # NCA специфичные параметры
                "activation": "tanh",  # Bounded activation для stability
                "dropout": 0.0,  # NCA обычно без dropout
                "use_memory": False,  # NCA имеет implicit memory
                "alpha_init": 0.1,  # Начальное значение update strength
                "beta_init": 0.05,  # Начальное значение neighbor influence
            },
            # НОВАЯ СЕКЦИЯ: Emergent Training Configuration
            "emergent_training": {
                # Base configuration (динамически привязанные)
                "teacher_model": "Meta-Llama-3-8B",
                "cube_dimensions": "{[xs, ys, zs]}",  # Список из lattice параметров
                # Emergent processing settings (КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ)
                "enable_full_cube_gradient": True,
                "spatial_propagation_depth": "{zs}",  # ИСПРАВЛЕНО: привязано к глубине решетки
                "emergent_specialization": True,
                # gMLP config будет заполнена из секции gmlp (специальная обработка)
                "gmlp_config": None,  # Будет заполнено в post-processing
                # Multi-objective loss configuration
                "loss_weights": {
                    "surface_reconstruction": 0.3,
                    "internal_consistency": 0.3,
                    "dialogue_similarity": 0.4,
                },
                # Training settings (привязанные к основной секции training)
                "learning_rate": "{learning_rate}",
                "batch_size": "{batch_size}",
                "epochs": "{epochs}",
                "warmup_epochs": 3,
                # Optimization settings
                "gradient_balancing": True,
                "adaptive_loss_weighting": True,
                "gradient_clip_norm": 1.0,
                "weight_decay": 0.01,
            },
        }

    def adjust_config_for_mode(
        self, config: Dict[str, Any], mode: str
    ) -> Dict[str, Any]:
        """Скорректировать конфигурацию для конкретного режима"""
        scale_factor = self.scale_settings.get_scale(mode)

        # Устанавливаем scale_factor
        config["lattice"]["scale_factor"] = scale_factor

        # Корректируем batch_size в зависимости от режима
        if mode == "development":
            config["training"]["batch_size"] = 16
        elif mode == "research":
            config["training"]["batch_size"] = 32
        elif mode == "validation":
            config["training"]["batch_size"] = 64
        elif mode == "production":
            config["training"]["batch_size"] = 128

        logger.info(f"[TARGET] Configured for {mode} mode (scale={scale_factor})")
        return config

    def generate_config(self, mode: str = "auto") -> Dict[str, Any]:
        """Сгенерировать полную конфигурацию для режима"""

        # Автоопределение режима если нужно
        if mode == "auto":
            mode = self.detect_hardware_mode()

        # Создаем базовый шаблон
        config = self.create_base_config_template()

        # Корректируем под режим
        config = self.adjust_config_for_mode(config, mode)

        # Вычисляем все выражения
        processed_config = self.evaluator.process_config_dict(config)

        # POST-PROCESSING: Специальная обработка для emergent_training
        if "emergent_training" in processed_config:
            # Проверяем какую архитектуру использовать
            use_nca = processed_config.get("nca", {}).get("enabled", False)

            if use_nca and "nca" in processed_config:
                # Используем NCA архитектуру
                processed_config["emergent_training"]["cell_architecture"] = "nca"
                processed_config["emergent_training"]["gmlp_config"] = processed_config[
                    "nca"
                ].copy()
                processed_config["emergent_training"]["nca_config"] = processed_config[
                    "nca"
                ].copy()
                logger.info(
                    "[POST-PROCESS] Using NCA architecture for emergent training"
                )
            elif "gmlp" in processed_config:
                # Fallback на gMLP архитектуру
                processed_config["emergent_training"]["cell_architecture"] = "gmlp"
                processed_config["emergent_training"]["gmlp_config"] = processed_config[
                    "gmlp"
                ].copy()
                logger.debug(
                    "[POST-PROCESS] Using gMLP architecture for emergent training"
                )
            else:
                logger.warning("[POST-PROCESS] No valid cell architecture found!")

        # Добавляем метаданные
        processed_config["_metadata"] = {
            "mode": mode,
            "scale_factor": processed_config["lattice"]["scale_factor"],
            "generated_by": "DynamicConfigGenerator",
            "bio_constants_version": "1.0",
        }

        logger.info(f"[OK] Generated config for {mode} mode:")
        logger.info(
            f"   Lattice: {processed_config['lattice']['xs']}x{processed_config['lattice']['ys']}x{processed_config['lattice']['zs']}"
        )

        # Безопасное форматирование чисел
        total_neurons = processed_config["lattice"]["total_neurons"]
        embedding_dim = processed_config["embeddings"]["embedding_dim"]

        if isinstance(total_neurons, (int, float)):
            logger.info(f"   Total neurons: {total_neurons:,}")
        else:
            logger.info(f"   Total neurons: {total_neurons}")

        if isinstance(embedding_dim, (int, float)):
            logger.info(f"   Embedding dim: {embedding_dim:,}")
        else:
            logger.info(f"   Embedding dim: {embedding_dim}")

        return processed_config


class DynamicConfigManager:
    """Основной менеджер динамической конфигурации"""

    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path("config")
        self.generator = DynamicConfigGenerator()

    def create_config_for_mode(self, mode: str) -> Dict[str, Any]:
        """Создать конфигурацию для конкретного режима"""
        return self.generator.generate_config(mode)

    def save_config(self, config: Dict[str, Any], filename: str) -> Path:
        """Сохранить конфигурацию в файл"""
        self.config_dir.mkdir(exist_ok=True)
        filepath = self.config_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"[SAVE] Saved config to {filepath}")
        return filepath

    def create_and_save_all_modes(self) -> Dict[str, Path]:
        """Создать и сохранить конфигурации для всех режимов"""
        modes = ["development", "research", "validation", "production"]
        saved_files = {}

        for mode in modes:
            config = self.create_config_for_mode(mode)
            filename = f"dynamic_config_{mode}.yaml"
            filepath = self.save_config(config, filename)
            saved_files[mode] = filepath

        return saved_files


# Удобные функции для использования
def generate_config_for_current_hardware() -> Dict[str, Any]:
    """Сгенерировать конфигурацию для текущего железа (автоопределение)"""
    manager = DynamicConfigManager()
    return manager.create_config_for_mode("auto")


def get_recommended_config() -> Dict[str, Any]:
    """Получить рекомендованную конфигурацию (алиас)"""
    return generate_config_for_current_hardware()


if __name__ == "__main__":
    # Демонстрация работы
    print("[BRAIN] Testing Dynamic Configuration System...")

    manager = DynamicConfigManager()

    # Тест автоопределения
    auto_config = manager.create_config_for_mode("auto")
    print(f"\n[TARGET] Auto-detected mode: {auto_config['_metadata']['mode']}")

    # Тест всех режимов
    for mode in ["development", "research", "validation"]:
        config = manager.create_config_for_mode(mode)
        lattice = config["lattice"]
        gmlp = config["gmlp"]
        print(f"\n[DATA] {mode.upper()} mode:")
        print(f"   Lattice: {lattice['xs']}x{lattice['ys']}x{lattice['zs']}")
        print(f"   Neurons: {lattice['total_neurons']:,}")
        print(f"   Batch: {config['training']['batch_size']}")
        print(f"   [BRAIN] gMLP target: {gmlp['target_params']} parameters")
        print(f"   state_size={gmlp['state_size']}, hidden_dim={gmlp['hidden_dim']}")
        print(
            f"   external_input={gmlp['external_input_size']}, memory={gmlp['memory_dim']}"
        )

    # Специальный тест с scale=0.06 (как в команде пользователя)
    print(f"\n[TARGET] SPECIAL TEST: Development mode with scale=0.06:")
    setattr(manager.generator.scale_settings, "development", 0.06)
    config_006 = manager.create_config_for_mode("development")
    lattice_006 = config_006["lattice"]
    gmlp_006 = config_006["gmlp"]
    print(f"   Lattice: {lattice_006['xs']}x{lattice_006['ys']}x{lattice_006['zs']}")
    print(f"   Neurons: {lattice_006['total_neurons']:,}")
    print(f"   [BRAIN] gMLP target: {gmlp_006['target_params']} parameters")
    print(
        f"   state_size={gmlp_006['state_size']}, hidden_dim={gmlp_006['hidden_dim']}"
    )
    print(
        f"   external_input={gmlp_006['external_input_size']}, memory={gmlp_006['memory_dim']}"
    )

    # Приблизительный расчет параметров
    state_size = gmlp_006["state_size"]
    hidden_dim = gmlp_006["hidden_dim"]
    ext_input = gmlp_006["external_input_size"]
    memory_dim = gmlp_006["memory_dim"]
    neighbor_count = 6

    # Примерный расчет (упрощенный для gMLP)
    input_size = neighbor_count * state_size + state_size + ext_input
    approx_params = (
        input_size * hidden_dim + hidden_dim * state_size + memory_dim * hidden_dim
    )
    print(
        f"   [DATA] Estimated gMLP params: ~{approx_params} (target: {gmlp_006['target_params']})"
    )
