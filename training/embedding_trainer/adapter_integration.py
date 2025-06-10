"""
[LINK] Universal Adapter Integration
Интеграция универсального адаптера с системой обучения CubeTrainer
Поддержка любых teacher моделей и размеров куба
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass
import logging
from pathlib import Path
import yaml

# Импорты нашей системы
from .cube_trainer import CubeTrainer, TrainingConfig, EmbeddingMetrics
from data.embedding_adapter.universal_adapter import (
    UniversalEmbeddingAdapter,
    AdapterManager,
    create_adapter_for_cube,
    KNOWN_MODELS,
)

# Условный импорт для EmbeddingProcessor.SURFACE_ONLY
try:
    from core.embedding_processor import (
        EmbeddingProcessor,
        create_surface_only_config,
        ProcessingMode,
    )

    EMBEDDING_PROCESSOR_AVAILABLE = True
except ImportError:
    # Fallback режим без EmbeddingProcessor
    EMBEDDING_PROCESSOR_AVAILABLE = False
    EmbeddingProcessor = None
    create_surface_only_config = None
    ProcessingMode = None

logger = logging.getLogger(__name__)


@dataclass
class AdapterIntegrationConfig:
    """Конфигурация интеграции адаптера с системой обучения"""

    # Teacher model settings
    teacher_model: str = "Meta-Llama-3-8B"
    teacher_embedding_dim: Optional[int] = None  # Auto-detect from KNOWN_MODELS

    # Cube settings
    cube_dimensions: Tuple[int, int, int] = (15, 15, 11)
    surface_strategy: str = "single"  # single | triple | full

    # Adapter settings
    adapter_strategy: str = (
        "learned_linear"  # learned_linear | hierarchical | attention_based | autoencoder
    )
    use_reconstruction_loss: bool = True
    reconstruction_weight: float = 0.1

    # Training settings
    adapter_learning_rate: float = 0.001
    cube_learning_rate: float = 0.0005
    joint_training: bool = True  # Обучаем adapter + cube совместно

    # Integration settings
    gradient_balancing: bool = True
    adapter_warmup_epochs: int = 3

    def __post_init__(self):
        if self.teacher_embedding_dim is None:
            # Сопоставляем наши ключи с известными моделями
            model_mapping = {
                "distilbert": "DistilBERT",
                "distilbert-base-uncased": "DistilBERT",
                "roberta": "RoBERTa-base",
                "roberta-base": "RoBERTa-base",
                "gpt2": "GPT-3.5",  # Приблизительно
                "dialogpt": "GPT-3.5",  # Приблизительно
                "llama3-8b-local": "Meta-Llama-3-8B",
                "llama3-8b": "Meta-Llama-3-8B",
                "meta-llama/meta-llama-3-8b": "Meta-Llama-3-8B",
            }

            # Проверяем прямое соответствие или через маппинг
            if self.teacher_model in KNOWN_MODELS:
                self.teacher_embedding_dim = KNOWN_MODELS[self.teacher_model][
                    "embedding_dim"
                ]
            elif self.teacher_model.lower() in model_mapping:
                mapped_model = model_mapping[self.teacher_model.lower()]
                self.teacher_embedding_dim = KNOWN_MODELS[mapped_model]["embedding_dim"]
                # Обновляем teacher_model для дальнейшего использования
                self.teacher_model = mapped_model
            else:
                # Пытаемся определить автоматически на основе имени модели
                if "distilbert" in self.teacher_model.lower():
                    self.teacher_embedding_dim = 768
                    self.teacher_model = "DistilBERT"
                elif "roberta" in self.teacher_model.lower():
                    self.teacher_embedding_dim = 768
                    self.teacher_model = "RoBERTa-base"
                elif "llama" in self.teacher_model.lower():
                    self.teacher_embedding_dim = 4096
                    self.teacher_model = "Meta-Llama-3-8B"
                elif "gpt" in self.teacher_model.lower():
                    self.teacher_embedding_dim = 1536
                    self.teacher_model = "GPT-3.5"
                else:
                    raise ValueError(
                        f"Unknown teacher model: {self.teacher_model}. "
                        f"Known models: {list(KNOWN_MODELS.keys())} or supported keys: {list(model_mapping.keys())}"
                    )


class AdapterCubeTrainer:
    """
    Интегрированный тренер с универсальным адаптером

    Комбинирует:
    - UniversalEmbeddingAdapter (любая модель → surface размер)
    - EmbeddingProcessor.SURFACE_ONLY (surface processing)
    - Joint training pipeline
    """

    def __init__(
        self,
        config: Optional[Union[AdapterIntegrationConfig, str, Dict]] = None,
        device: str = "cpu",
    ):
        """
        Инициализация интегрированного тренера

        Args:
            config: Конфигурация интеграции
            device: Устройство для вычислений
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("[LINK] Initializing AdapterCubeTrainer...")

        # Загрузка конфигурации
        self.config = self._load_config(config)
        self.device = torch.device(device)

        # Компоненты системы
        self.adapter = None
        self.embedding_processor = (
            None  # Заменяем cube_trainer на direct EmbeddingProcessor
        )
        self.adapter_optimizer = None
        self.joint_optimizer = None

        # Состояние обучения
        self.current_epoch = 0
        self.training_history = []
        self.adapter_warmup_complete = False

        self.logger.info(f"[OK] AdapterCubeTrainer configured:")
        self.logger.info(
            f"   Teacher: {self.config.teacher_model} ({self.config.teacher_embedding_dim}D)"
        )
        self.logger.info(f"   Cube: {self.config.cube_dimensions}")
        self.logger.info(f"   Surface strategy: {self.config.surface_strategy}")
        self.logger.info(f"   Adapter strategy: {self.config.adapter_strategy}")

    def _load_config(
        self, config: Optional[Union[AdapterIntegrationConfig, str, Dict]]
    ) -> AdapterIntegrationConfig:
        """Загрузка конфигурации интеграции"""
        if config is None:
            return AdapterIntegrationConfig()

        elif isinstance(config, AdapterIntegrationConfig):
            return config

        elif isinstance(config, str):
            # Загрузка из YAML
            try:
                with open(config, "r", encoding="utf-8") as f:
                    config_data = yaml.safe_load(f)
                return AdapterIntegrationConfig(**config_data)
            except Exception as e:
                self.logger.warning(f"Failed to load config from {config}: {e}")
                return AdapterIntegrationConfig()

        elif isinstance(config, dict):
            try:
                return AdapterIntegrationConfig(**config)
            except Exception as e:
                self.logger.warning(f"Failed to create config from dict: {e}")
                return AdapterIntegrationConfig()

        else:
            return AdapterIntegrationConfig()

    def initialize_components(self):
        """Инициализация всех компонентов интегрированной системы"""
        self.logger.info("[CONFIG] Initializing integrated training components...")

        # 1. Создание универсального адаптера
        self._setup_adapter()

        # 2. Настройка EmbeddingProcessor (заменяет CubeTrainer)
        self._setup_embedding_processor()

        # 3. Настройка оптимизаторов
        self._setup_optimizers()

        # 4. Проверка совместимости
        self._validate_integration()

        self.logger.info("[TARGET] All integrated components initialized successfully!")

    def _setup_adapter(self):
        """Создание и настройка универсального адаптера"""
        self.logger.info(
            f"[CONFIG] Setting up adapter: {self.config.teacher_model} → surface..."
        )

        # Вычисляем surface размер в зависимости от стратегии
        surface_size = self._calculate_surface_size()

        # Создаем адаптер
        self.adapter = UniversalEmbeddingAdapter(
            input_dim=self.config.teacher_embedding_dim,
            output_dim=surface_size,
            strategy=self.config.adapter_strategy,
        )

        self.adapter.to(self.device)

        compression_ratio = self.adapter.get_compression_ratio()
        param_count = self.adapter.get_parameter_count()

        self.logger.info(f"[OK] Adapter created:")
        self.logger.info(f"   {self.config.teacher_embedding_dim}D → {surface_size}D")
        self.logger.info(
            f"   Compression: {compression_ratio:.3f} ({compression_ratio*100:.1f}%)"
        )
        self.logger.info(f"   Parameters: {param_count:,}")

    def _setup_embedding_processor(self):
        """Настройка EmbeddingProcessor.SURFACE_ONLY (заменяет SimpleWrapper)"""
        if not EMBEDDING_PROCESSOR_AVAILABLE:
            self.logger.warning(
                "[WARNING] EmbeddingProcessor not available, using simple identity processor"
            )
            self.embedding_processor = self._create_simple_processor()
            return

        self.logger.info("[CONFIG] Setting up EmbeddingProcessor.SURFACE_ONLY...")

        # Вычисляем параметры surface processing
        surface_size = self._calculate_surface_size()
        surface_dims = self.config.cube_dimensions[:2]  # (15, 15)

        # Создаем surface-only конфигурацию
        processor_config = create_surface_only_config(
            surface_size=surface_size, surface_dims=surface_dims
        )

        # Настройка устройства
        processor_config.device = str(self.device)

        # Создаем EmbeddingProcessor напрямую
        self.embedding_processor = EmbeddingProcessor(processor_config)
        self.embedding_processor.to(self.device)

        self.logger.info(f"[OK] EmbeddingProcessor.SURFACE_ONLY initialized:")
        self.logger.info(f"   Mode: {ProcessingMode.SURFACE_ONLY.value}")
        self.logger.info(f"   Surface size: {surface_size}D")
        self.logger.info(f"   Surface dims: {surface_dims}")
        self.logger.info(
            f"   Processing depth: {processor_config.surface_processing_depth}"
        )

    def _create_simple_processor(self):
        """Создает простой identity processor как fallback"""
        import torch.nn as nn

        class SimpleIdentityProcessor(nn.Module):
            def __init__(self):
                super().__init__()
                # Добавляем dummy parameter чтобы оптимизатор не ломался
                self.dummy_param = nn.Parameter(torch.zeros(1), requires_grad=False)

            def forward(self, x):
                if isinstance(x, dict) and "processed_embeddings" in x:
                    return x  # Уже обработанный результат
                return {"processed_embeddings": x, "surface_output": x}

        return SimpleIdentityProcessor()

    def _setup_optimizers(self):
        """Настройка оптимизаторов для joint training"""
        self.logger.info("[CONFIG] Setting up optimizers...")

        # Всегда создаем все оптимизаторы для гибкости переключения режимов
        self.adapter_optimizer = optim.Adam(
            self.adapter.parameters(), lr=self.config.adapter_learning_rate
        )

        # Создаем отдельный optimizer для EmbeddingProcessor
        processor_params = list(self.embedding_processor.parameters())

        if len(processor_params) > 0:
            self.processor_optimizer = optim.AdamW(
                processor_params, lr=self.config.cube_learning_rate
            )
            self.logger.info(
                f"[OK] Processor optimizer created with {len(processor_params)} parameters"
            )
        else:
            self.processor_optimizer = None
            self.logger.info("[WARNING] Processor optimizer skipped (no trainable parameters)")

        if self.config.joint_training:
            # Joint optimizer для обоих компонентов
            all_params = list(self.adapter.parameters())
            if processor_params:
                all_params.extend(processor_params)

            if len(all_params) > 0:
                self.joint_optimizer = optim.AdamW(
                    all_params, lr=self.config.cube_learning_rate
                )
                self.logger.info(
                    f"[OK] Joint optimizer configured with {len(all_params)} total parameters"
                )
            else:
                self.joint_optimizer = None
                self.logger.warning("[WARNING] Joint optimizer creation failed: no parameters")
        else:
            self.joint_optimizer = None

        self.logger.info("[OK] All optimizers configured (adapter, processor, joint)")

    def _calculate_surface_size(self) -> int:
        """Вычисление размера surface в зависимости от стратегии"""
        x, y, z = self.config.cube_dimensions
        face_size = x * y

        if self.config.surface_strategy == "single":
            return face_size
        elif self.config.surface_strategy == "triple":
            return 3 * face_size  # front, back, top
        elif self.config.surface_strategy == "full":
            return 6 * face_size  # all faces
        else:
            raise ValueError(
                f"Unknown surface strategy: {self.config.surface_strategy}"
            )

    def _validate_integration(self):
        """Проверка совместимости всех компонентов"""
        self.logger.info("[MAGNIFY] Validating component integration...")

        # Тестовый forward pass
        test_input = torch.randn(2, self.config.teacher_embedding_dim).to(self.device)

        # Adapter forward
        adapter_output = self.adapter(test_input)
        expected_surface_size = self._calculate_surface_size()

        assert adapter_output.shape == (
            2,
            expected_surface_size,
        ), f"Adapter output shape mismatch: {adapter_output.shape} vs expected (2, {expected_surface_size})"

        # EmbeddingProcessor forward
        processor_result = self.embedding_processor.forward(adapter_output)

        # Handle case when processor returns dict
        if isinstance(processor_result, dict):
            processor_output = processor_result.get(
                "processed_embeddings",
                processor_result.get("surface_output", adapter_output),
            )
        else:
            processor_output = processor_result

        assert (
            processor_output.shape == adapter_output.shape
        ), f"Processor output shape mismatch: {processor_output.shape} vs expected {adapter_output.shape}"

        self.logger.info("[OK] Integration validation passed!")
        self.logger.info(
            f"   Pipeline: {test_input.shape} → {adapter_output.shape} → {processor_output.shape}"
        )

    def forward(
        self, teacher_embeddings: torch.Tensor, return_intermediate: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Full forward pass через adapter + EmbeddingProcessor.SURFACE_ONLY

        Args:
            teacher_embeddings: Эмбединги от teacher модели [batch, teacher_dim]
            return_intermediate: Возвращать промежуточные результаты

        Returns:
            output: Финальный результат [batch, surface_size]
            intermediate: Словарь промежуточных результатов (если запрошено)
        """
        # 1. Adapter: teacher_dim → surface_size
        if self.config.use_reconstruction_loss:
            surface_embeddings, reconstructed = self.adapter(
                teacher_embeddings, return_reconstruction=True
            )
        else:
            surface_embeddings = self.adapter(teacher_embeddings)
            reconstructed = None

        # 2. EmbeddingProcessor.SURFACE_ONLY: surface_size → surface_size
        processed_result = self.embedding_processor.forward(surface_embeddings)

        # Handle case when processor returns dict
        if isinstance(processed_result, dict):
            processed_embeddings = processed_result.get(
                "processed_embeddings",
                processed_result.get("surface_output", surface_embeddings),
            )
        else:
            processed_embeddings = processed_result

        if return_intermediate:
            return {
                "output": processed_embeddings,
                "surface_embeddings": surface_embeddings,
                "reconstructed": reconstructed,
                "teacher_embeddings": teacher_embeddings,
            }
        else:
            return processed_embeddings

    def compute_loss(
        self, teacher_embeddings: torch.Tensor, target_embeddings: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Вычисление комбинированной loss function

        Args:
            teacher_embeddings: Input эмбединги от teacher модели
            target_embeddings: Target эмбединги (также от teacher модели)

        Returns:
            Dict с различными компонентами loss
        """
        # Forward pass с промежуточными результатами
        results = self.forward(teacher_embeddings, return_intermediate=True)

        # Конвертируем target также через adapter
        target_surface = self.adapter(target_embeddings)

        # Main task loss: processed surface → target surface
        main_loss = nn.functional.cosine_embedding_loss(
            results["output"],
            target_surface,
            torch.ones(target_surface.size(0)).to(self.device),
        )

        losses = {"main_loss": main_loss}

        # Reconstruction loss (если включен)
        if self.config.use_reconstruction_loss and results["reconstructed"] is not None:
            reconstruction_loss = self.adapter.compute_reconstruction_loss(
                teacher_embeddings, results["reconstructed"]
            )
            losses["reconstruction_loss"] = reconstruction_loss

        # Total loss
        total_loss = losses["main_loss"]
        if "reconstruction_loss" in losses:
            total_loss = (
                total_loss
                + self.config.reconstruction_weight * losses["reconstruction_loss"]
            )

        losses["total_loss"] = total_loss

        return losses

    def train_step(
        self, question_embeddings: torch.Tensor, answer_embeddings: torch.Tensor
    ) -> Dict[str, float]:
        """
        Один шаг обучения

        Args:
            question_embeddings: Question эмбединги от teacher модели
            answer_embeddings: Answer эмбединги от teacher модели

        Returns:
            Метрики шага обучения
        """
        if self.config.joint_training:
            return self._joint_train_step(question_embeddings, answer_embeddings)
        else:
            return self._separate_train_step(question_embeddings, answer_embeddings)

    def _joint_train_step(
        self, question_embeddings: torch.Tensor, answer_embeddings: torch.Tensor
    ) -> Dict[str, float]:
        """Joint training step (adapter + EmbeddingProcessor одновременно)"""
        self.adapter.train()
        self.embedding_processor.train()

        # Forward pass
        losses = self.compute_loss(question_embeddings, answer_embeddings)

        # Backward pass
        self.joint_optimizer.zero_grad()
        losses["total_loss"].backward()

        # Gradient clipping
        if self.config.gradient_balancing:
            nn.utils.clip_grad_norm_(self.adapter.parameters(), max_norm=1.0)
            nn.utils.clip_grad_norm_(
                self.embedding_processor.parameters(), max_norm=1.0
            )

        self.joint_optimizer.step()

        # Metrics
        with torch.no_grad():
            results = self.forward(question_embeddings, return_intermediate=True)
            target_surface = self.adapter(answer_embeddings)

            qa_similarity = (
                torch.cosine_similarity(results["output"], target_surface, dim=1)
                .mean()
                .item()
            )

        return {
            "total_loss": losses["total_loss"].item(),
            "main_loss": losses["main_loss"].item(),
            "reconstruction_loss": losses.get(
                "reconstruction_loss", torch.tensor(0.0)
            ).item(),
            "qa_similarity": qa_similarity,
        }

    def _separate_train_step(
        self, question_embeddings: torch.Tensor, answer_embeddings: torch.Tensor
    ) -> Dict[str, float]:
        """Separate training (adapter first, then EmbeddingProcessor)"""
        if (
            not self.adapter_warmup_complete
            and self.current_epoch < self.config.adapter_warmup_epochs
        ):
            # Adapter warmup phase
            return self._adapter_warmup_step(question_embeddings, answer_embeddings)
        else:
            # Main training phase
            self.adapter_warmup_complete = True
            return self._processor_training_step(question_embeddings, answer_embeddings)

    def _adapter_warmup_step(
        self, question_embeddings: torch.Tensor, answer_embeddings: torch.Tensor
    ) -> Dict[str, float]:
        """Adapter warmup step (только adapter)"""
        self.adapter.train()
        self.embedding_processor.eval()  # Freeze EmbeddingProcessor

        # Adapter training только на reconstruction
        _, reconstructed = self.adapter(question_embeddings, return_reconstruction=True)
        reconstruction_loss = self.adapter.compute_reconstruction_loss(
            question_embeddings, reconstructed
        )

        self.adapter_optimizer.zero_grad()
        reconstruction_loss.backward()
        self.adapter_optimizer.step()

        return {
            "total_loss": reconstruction_loss.item(),
            "reconstruction_loss": reconstruction_loss.item(),
            "qa_similarity": 0.0,  # Not applicable during warmup
            "phase": "adapter_warmup",
        }

    def _processor_training_step(
        self, question_embeddings: torch.Tensor, answer_embeddings: torch.Tensor
    ) -> Dict[str, float]:
        """EmbeddingProcessor training step (только EmbeddingProcessor, adapter frozen)"""
        self.adapter.eval()  # Freeze adapter
        self.embedding_processor.train()

        # Forward через frozen adapter
        with torch.no_grad():
            question_surface = self.adapter(question_embeddings)
            answer_surface = self.adapter(answer_embeddings)

        # EmbeddingProcessor training
        processed_surface = self.embedding_processor.forward(question_surface)

        # Handle case when processor returns dict
        if isinstance(processed_surface, dict):
            processed_surface = processed_surface.get(
                "processed_embeddings",
                processed_surface.get("surface_output", processed_surface),
            )

        main_loss = nn.functional.cosine_embedding_loss(
            processed_surface,
            answer_surface,
            torch.ones(answer_surface.size(0)).to(self.device),
        )

        # Backward pass (только если есть trainable parameters)
        if self.processor_optimizer is not None:
            self.processor_optimizer.zero_grad()
            main_loss.backward()
            self.processor_optimizer.step()
        else:
            # Если нет trainable parameters, loss все равно нужен для метрик
            pass

        # Metrics
        with torch.no_grad():
            qa_similarity = (
                torch.cosine_similarity(processed_surface, answer_surface, dim=1)
                .mean()
                .item()
            )

        return {
            "total_loss": main_loss.item(),
            "main_loss": main_loss.item(),
            "qa_similarity": qa_similarity,
            "phase": "processor_training",
        }

    def get_info(self) -> Dict[str, Any]:
        """Информация о интегрированной системе"""
        adapter_info = {
            "input_dim": self.adapter.input_dim if self.adapter else None,
            "output_dim": self.adapter.output_dim if self.adapter else None,
            "strategy": self.config.adapter_strategy,
            "compression_ratio": (
                self.adapter.get_compression_ratio() if self.adapter else None
            ),
            "parameters": self.adapter.get_parameter_count() if self.adapter else None,
        }

        processor_info = {
            "type": "EmbeddingProcessor.SURFACE_ONLY",
            "mode": ProcessingMode.SURFACE_ONLY.value,
            "surface_size": self._calculate_surface_size(),
            "surface_dims": self.config.cube_dimensions[:2],
            "total_parameters": (
                sum(p.numel() for p in self.embedding_processor.parameters())
                if self.embedding_processor
                else 0
            ),
        }

        return {
            "teacher_model": self.config.teacher_model,
            "cube_dimensions": self.config.cube_dimensions,
            "surface_strategy": self.config.surface_strategy,
            "joint_training": self.config.joint_training,
            "current_epoch": self.current_epoch,
            "adapter_warmup_complete": self.adapter_warmup_complete,
            "adapter": adapter_info,
            "processor": processor_info,
            "total_parameters": (
                adapter_info.get("parameters", 0)
                + processor_info.get("total_parameters", 0)
            ),
        }


# Удобные функции для создания интегрированных тренеров


def create_llama3_cube_trainer(
    cube_dimensions: Tuple[int, int, int] = (15, 15, 11),
    adapter_strategy: str = "learned_linear",
    device: str = "cpu",
) -> AdapterCubeTrainer:
    """Создание тренера для LLaMA-3-8B"""
    config = AdapterIntegrationConfig(
        teacher_model="Meta-Llama-3-8B",
        cube_dimensions=cube_dimensions,
        adapter_strategy=adapter_strategy,
        joint_training=True,
    )

    trainer = AdapterCubeTrainer(config=config, device=device)
    trainer.initialize_components()

    return trainer


def create_distilbert_cube_trainer(
    cube_dimensions: Tuple[int, int, int] = (15, 15, 11),
    adapter_strategy: str = "learned_linear",
    device: str = "cpu",
) -> AdapterCubeTrainer:
    """Создание тренера для DistilBERT (для сравнения)"""
    config = AdapterIntegrationConfig(
        teacher_model="DistilBERT",
        cube_dimensions=cube_dimensions,
        adapter_strategy=adapter_strategy,
        joint_training=True,
    )

    trainer = AdapterCubeTrainer(config=config, device=device)
    trainer.initialize_components()

    return trainer


# Экспорт
__all__ = [
    "AdapterIntegrationConfig",
    "AdapterCubeTrainer",
    "create_llama3_cube_trainer",
    "create_distilbert_cube_trainer",
]
