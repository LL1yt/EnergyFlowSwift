# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

This is a research project implementing a **3D Cellular Neural Network** inspired by biological brain structures. The system uses cellular automata-like structures arranged in a 3D lattice where each cell runs the same neural network prototype but processes signals from neighboring cells.

### Контекст проекта

- Исследовательский проект на одного разработчика
- Структура: AA/ (легаси), AA/new_rebuild/ (активная разработка), AA/archive/ (старые версии)
- Тесты запускаются из корня AA/
- new_rebuild/Рабочие рещения из Legacy проекта, которые можно использовать.md - можно использовать как примеры

### Структура `new_rebuild`

Директория `new_rebuild` содержит основную, переработанную архитектуру проекта.

- **`new_rebuild/config/`**: Содержит централизованную конфигурацию проекта (`project_config.py`).
- **`new_rebuild/core/`**: Ядро архитектуры, включающее:
  - **`cells/`**: Определения различных типов клеток.
  - **`cnf/`**: Компоненты для Continuous Normalizing Flows.
  - **`lattice/`**: Реализация 3D-решетки, пространственного хеширования и оптимизаций.
  - **`moe/`**: Компоненты Mixture of Experts.
- **`new_rebuild/utils/`**: Вспомогательные утилиты, такие как управление устройствами (`device_manager.py`) и логирование (`logging.py`).

### Принципы работы

**приоритет на скорость**

- Минимальные церемонии, максимальная эффективность, в том смысле, что можно пожертвовать перфекционизмом в угоду простому и эффективному решению
- Используй современные языковые возможности
- Предпочитай прямолинейные решения сложным абстракциям

при этом нас НЕ интересует продакшн и все что с этим связано, когда речь идет о больших коммерческих проектах с большим числом сотрудников
