emergent_training переименовать в -> training (так же Реализация processing концепции: TRAINING MODE: размер эмбединга обучающей llm → размер входного эмбединга поверхности куба → FULL CUBE INFLUENCE → размер выходного эмбединга поверхности куба → Learning; INFERENCE MODE: Question → размер входного эмбединга поверхности куба Front → [PROCESSING] → размер выходного эмбединга поверхности куба Back → Answer)
training\automated_training - можно использовать, как основу, но убрать все CLI и постараться реализовать обучными средствами атоматизацию.
production_training
inference\lightweight_decoder Компактный декодер для преобразования эмбедингов в текст
data\embedding_loader - Модуль для загрузки и предобработки векторных представлений (эмбедингов) различных типов. Обеспечивает унифицированный интерфейс для работы с популярными форматами эмбедингов в контексте 3D клеточной нейронной сети.
data\embeddings - готовые эмбединги от DistilBERT
training\embedding_trainer\dialogue_dataset.py - DialogueDataset - Класс для подготовки данных к обучению куба в dialogue режиме Этот модуль реализует специализированный dataset для обучения 3D Cubic Core на задачах диалога (question_embedding → answer_embedding).
training\embedding_trainer\autoencoder_dataset.py - AutoencoderDataset - Класс для подготовки данных к обучению куба в autoencoder режиме Этот модуль реализует специализированный dataset для обучения 3D Cubic Core на задачах реконструкции эмбедингов (autoencoder mode).
training\embedding_trainer\advanced_loss_functions.py - Продвинутая система loss functions для Stage 2.3 Включает: - Curriculum learning loss (easy→hard progression) - Triplet loss для enhanced semantic alignment - Contrastive learning approaches - Multi-objective optimization (similarity + diversity)
training\embedding_trainer\neural_cellular_automata.py - Реализация emergent behavior preservation во время GPU-optimized training. Ключевые принципы NCA для 3D Cellular Neural Network: 1. Stochastic Cell Updates - избежание глобальной синхронизации 2. Residual Update Rules - маленькие, стабильные модификации 3. Pattern Formation Metrics - количественная оценка emergence 4. Emergent Behavior Preservation - сохранение паттернов при оптимизации
download_distilbert.py - Скрипт для предварительной загрузки DistilBERT в локальную папку проекта models/local_cache.
generate_large_embedding_dataset.py - Генератор большого датасета эмбеддингов для обучения 3D куба Создает тысячи пар question-answer и сохраняет готовые эмбеддинги
generate_snli_embedding_dataset.py - Генератор эмбеддингов из SNLI датасета для обучения 3D куба Использует 1/5 часть SNLI (Stanford Natural Language Inference) датасета
precomputed_embedding_loader.py - Загрузчик готовых эмбеддингов из предварительно сгенерированного файла Используется для быстрого обучения без пересчета эмбеддингов
study_plan реализации архитектуры на основе локальных правил и эмерджентной связности.md - это последняя попытка интегрировать новую архитектуру в проект, но она завершилась неудачей.
Современные методы динамической связности для крупномасштабных 3D клеточных нейронных сетей.md - понимание оптимизации новой архитектуры
PHASE_5_PLUS_ROADMAP.md
