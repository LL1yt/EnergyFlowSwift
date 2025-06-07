# 📚 DOCUMENTATION INDEX - 3D Cellular Neural Network Project

**Цель:** Полная навигация по всей документации проекта с четкой иерархией.  
**Последнее обновление:** 6 декабря 2024  
**Статус проекта:** 82% завершен, Phase 2.7 Stage 2.3 активный

---

## 🎯 БЫСТРАЯ НАВИГАЦИЯ

### **🏗️ ГЛАВНЫЕ ПЛАНЫ (Master Level)**

- **[`PROJECT_PLAN.md`](PROJECT_PLAN.md)** - 🎯 **ГЛАВНЫЙ ПЛАН ПРОЕКТА** (начни здесь!)
- **[`PHASE_1_PLAN.md`](PHASE_1_PLAN.md)** - Foundation (завершен ✅)
- **[`PHASE_2_PLAN.md`](PHASE_2_PLAN.md)** - Core Functionality (завершен ✅)
- **[`PHASE_3_PLAN.md`](PHASE_3_PLAN.md)** - Training Infrastructure (планируется)

### **🎯 СПЕЦИАЛИЗИРОВАННЫЕ ПЛАНЫ (Phase Level)**

- **[`PHASE_2_3_PLAN.md`](PHASE_2_3_PLAN.md)** - EmbeddingReshaper (завершен ✅)
- **[`PHASE_2_5_PLAN.md`](PHASE_2_5_PLAN.md)** - Phrase Architecture (завершен ✅)
- **[`PHASE_2_7_PLAN.md`](PHASE_2_7_PLAN.md)** - Lightweight Decoder (90% завершен 🚀)

### **⚙️ МОДУЛЬНЫЕ ПЛАНЫ (Module Level)**

- **[`core/lattice_3d/plan.md`](core/lattice_3d/plan.md)** - 3D Cubic Core
- **[`data/embedding_loader/plan.md`](data/embedding_loader/plan.md)** - Teacher LLM Encoder
- **[`inference/lightweight_decoder/plan.md`](inference/lightweight_decoder/plan.md)** - Lightweight Decoder

### **🔧 CURSOR GUIDELINES**

- **[`.cursor/rules/plan-guidelines.mdc`](.cursor/rules/plan-guidelines.mdc)** - План иерархии
- **[`.cursor/rules/documentation-guidelines.mdc`](.cursor/rules/documentation-guidelines.mdc)** - Управление документацией

---

## 📋 ПОЛНАЯ СТРУКТУРА ДОКУМЕНТАЦИИ

### **🏗️ УРОВЕНЬ 1: ПРОЕКТНЫЕ ПЛАНЫ**

#### **Основные планы проекта:**

```
PROJECT_PLAN.md              # 🎯 MASTER PLAN - общий обзор проекта
├── PHASE_1_PLAN.md          # Foundation (завершен ✅)
├── PHASE_2_PLAN.md          # Core Functionality (завершен ✅)
├── PHASE_2_3_PLAN.md        # EmbeddingReshaper (завершен ✅)
├── PHASE_2_5_PLAN.md        # Phrase Architecture (завершен ✅)
├── PHASE_2_7_PLAN.md        # Lightweight Decoder (активный 🚀)
└── PHASE_3_PLAN.md          # Training Infrastructure (планируется)
```

#### **Исследовательские документы:**

```
GENERATIVE_DECODER_RESEARCH_SUMMARY.md           # Архитектурные исследования
ARCHITECTURE_RECOMMENDATIONS_ANALYSIS.md         # Топ-3 рекомендации 2024
IMPLEMENTATION_STRATEGY_V3.md                    # Стратегия реализации v3
```

---

### **⚙️ УРОВЕНЬ 2: МОДУЛЬНАЯ ДОКУМЕНТАЦИЯ**

#### **Модуль 1: Teacher LLM Encoder (ЗАВЕРШЕН ✅)**

```
data/embedding_loader/
├── README.md                # Описание и использование
├── plan.md                  # Детальный план реализации
├── meta.md                  # Exports и зависимости
├── errors.md                # Реальные ошибки разработки
├── diagram.mmd              # Архитектурная диаграмма
└── examples.md              # Примеры использования
```

#### **Модуль 2: 3D Cubic Core (ЗАВЕРШЕН ✅)**

```
core/lattice_3d/
├── README.md                # 3D решетка клеток
├── plan.md                  # План разработки ядра
├── meta.md                  # Экспорты и API
├── errors.md                # Документированные проблемы
├── diagram.mmd              # Диаграмма архитектуры
└── examples.md              # Примеры использования

core/embedding_processor/
├── README.md                # Процессор эмбедингов
├── plan.md                  # План интеграции
├── meta.md                  # Технические спецификации
├── errors.md                # Ошибки интеграции
├── diagram.mmd              # Схема обработки
└── examples.md              # Примеры эмбединг обработки
```

#### **Модуль 3: Lightweight Decoder (90% ЗАВЕРШЕН 🚀)**

```
inference/lightweight_decoder/
├── README.md                            # Production документация
├── plan.md                              # 🎯 АКТИВНЫЙ ПЛАН (Stage 2.3)
├── meta.md                              # API спецификации
├── errors.md                            # Проблемы разработки
├── diagram.mmd                          # Архитектура декодера
├── examples.md                          # Примеры декодирования
├── STAGE_2_1_COMPLETION_REPORT.md      # Отчет Stage 2.1 (RET v2.1)
└── STAGE_2_2_COMPLETION_REPORT.md      # Отчет Stage 2.2 (Integration)
```

---

### **🔧 УРОВЕНЬ 3: ТЕХНИЧЕСКИЕ ДОКУМЕНТЫ**

#### **Cursor Guidelines & Rules:**

```
.cursor/rules/
├── plan-guidelines.mdc                  # 📋 Иерархия планирования
├── documentation-guidelines.mdc         # 📚 Управление документацией
└── project-guidelines.mdc               # 🎯 Общие правила проекта
```

#### **Конфигурация:**

```
config/
├── main_config.yaml                     # Основная конфигурация
├── lightweight_decoder.yaml             # Конфигурация декодера v3.0.0
└── phase_config.yaml                    # Настройки фаз
```

#### **Основные файлы:**

```
README.md                                # Описание проекта
instructions.md                          # Полные инструкции разработки
requirements.txt                         # Python зависимости
main.py                                  # Точка интеграции модулей
```

---

## 🎯 НАВИГАЦИЯ ПО СТАТУСУ

### **✅ ЗАВЕРШЕННЫЕ МОДУЛИ (готовы к использованию):**

**Teacher LLM Encoder (Модуль 1):**

- **Документация:** [`data/embedding_loader/README.md`](data/embedding_loader/README.md)
- **План:** [`data/embedding_loader/plan.md`](data/embedding_loader/plan.md)
- **Статус:** Production-ready, 8+ LLM моделей

**3D Cubic Core (Модуль 2):**

- **Документация:** [`core/lattice_3d/README.md`](core/lattice_3d/README.md)
- **План:** [`core/lattice_3d/plan.md`](core/lattice_3d/plan.md)
- **Статус:** 0.999 cosine similarity, production-ready

**EmbeddingReshaper:**

- **Документация:** [`data/embedding_reshaper/README.md`](data/embedding_reshaper/README.md)
- **План:** [`PHASE_2_3_PLAN.md`](PHASE_2_3_PLAN.md)
- **Статус:** 100% семантическое сохранение

**PhraseBankDecoder:**

- **Документация:** [`inference/lightweight_decoder/README.md`](inference/lightweight_decoder/README.md)
- **Статус:** 17/17 тестов пройдено, production-ready

### **🚀 АКТИВНЫЕ МОДУЛИ (в разработке):**

**GenerativeDecoder (Stage 2.3):**

- **Активный план:** [`inference/lightweight_decoder/plan.md`](inference/lightweight_decoder/plan.md)
- **Статус:** RET v2.1 integration complete (16/16 тестов)
- **Следующий шаг:** Quality optimization & training preparation

### **💡 ПЛАНИРУЕМЫЕ МОДУЛИ:**

**Phase 3 Training Infrastructure:**

- **План:** [`PHASE_3_PLAN.md`](PHASE_3_PLAN.md) (будет создан)
- **Статус:** Готов к запуску после завершения Phase 2.7

---

## 📊 АКТУАЛЬНЫЕ МЕТРИКИ ПРОЕКТА

### **Общий прогресс: 82% 🚀**

- **Phase 1:** ✅ 100% (Foundation)
- **Phase 2:** ✅ 100% (Core Functionality + extensions)
- **Phase 2.7:** 🚀 90% (Lightweight Decoder)
  - Stage 1: ✅ 100% (PhraseBankDecoder - 17/17 тестов)
  - Stage 2.1: ✅ 100% (RET v2.1 - 8/8 тестов)
  - Stage 2.2: ✅ 100% (Integration - 8/8 тестов)
  - Stage 2.3: 🎯 Активный (Quality optimization)
- **Phase 3:** 💡 0% (Training Infrastructure - готов к запуску)

### **Модульная готовность:**

- **🔴 Модуль 1 (Teacher LLM Encoder):** ✅ 100%
- **🔵 Модуль 2 (3D Cubic Core):** ✅ 100%
- **🟡 Модуль 3 (Lightweight Decoder):** 🚀 90%

### **Покрытие тестами: 34/34 пройдено (100%)**

- **Phase 1 tests:** 8/8 ✅
- **Phase 2 tests:** 10/10 ✅
- **PhraseBankDecoder:** 17/17 ✅
- **GenerativeDecoder:** 16/16 ✅ (Stage 2.1 + 2.2)

---

## 🔍 КАК НАЙТИ НУЖНУЮ ИНФОРМАЦИЮ

### **Если нужно понять общую архитектуру:**

1. Начни с [`PROJECT_PLAN.md`](PROJECT_PLAN.md)
2. Изучи модульную архитектуру
3. Перейди к конкретным планам фаз

### **Если работаешь с конкретным модулем:**

1. Открой `module/README.md` для общего понимания
2. Изучи `module/plan.md` для детального планирования
3. Посмотри `module/examples.md` для примеров использования

### **Если нужно обновить планы:**

1. Изучи [`.cursor/rules/plan-guidelines.mdc`](.cursor/rules/plan-guidelines.mdc)
2. Следуй иерархии: module → phase → project
3. Используй [`.cursor/rules/documentation-guidelines.mdc`](.cursor/rules/documentation-guidelines.mdc)

### **Если нужна актуальная информация о прогрессе:**

1. [`PROJECT_PLAN.md`](PROJECT_PLAN.md) - общий статус
2. [`PHASE_2_7_PLAN.md`](PHASE_2_7_PLAN.md) - текущая активная фаза
3. [`inference/lightweight_decoder/plan.md`](inference/lightweight_decoder/plan.md) - детали Stage 2.3

---

## 🚀 РЕКОМЕНДУЕМЫЕ ПУТИ ЧТЕНИЯ

### **Для нового разработчика:**

```
1. README.md                              # Общее понимание
2. PROJECT_PLAN.md                        # Архитектура и прогресс
3. instructions.md                        # Правила разработки
4. .cursor/rules/plan-guidelines.mdc      # Система планирования
5. PHASE_2_7_PLAN.md                      # Текущая работа
```

### **Для работы с конкретным модулем:**

```
1. module/README.md                       # Понимание модуля
2. module/plan.md                         # Текущие задачи
3. module/examples.md                     # Примеры использования
4. module/meta.md                         # Технические детали
```

### **Для планирования новой фичи:**

```
1. PROJECT_PLAN.md                        # Общий контекст
2. PHASE_X_PLAN.md                        # Фазовое планирование
3. .cursor/rules/plan-guidelines.mdc      # Правила планирования
4. module/plan.md                         # Модульные детали
```

---

**🎯 ПРИНЦИП НАВИГАЦИИ: "От общего к частному, с пониманием иерархии"**

_Начинай с общих планов, углубляйся в детали по мере необходимости, всегда поддерживай актуальность документации._
