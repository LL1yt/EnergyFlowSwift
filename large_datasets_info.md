# Большие готовые датасеты для обучения эмбеддингов

## 1. Hugging Face Datasets

### MS Marco Dataset

- **Размер**: ~8.8M пар question-passage
- **Источник**: `microsoft/ms_marco`
- **Описание**: Вопросы из поисковых запросов Bing + релевантные отрывки
- **Качество**: Высокое, реальные поисковые запросы
- **Загрузка**: `datasets.load_dataset('ms_marco', 'v1.1')`

### Natural Questions

- **Размер**: ~300K вопросов
- **Источник**: `natural_questions`
- **Описание**: Вопросы из Google Search + ответы из Wikipedia
- **Качество**: Очень высокое, проверенные экспертами
- **Загрузка**: `datasets.load_dataset('natural_questions')`

### SQuAD 2.0

- **Размер**: ~150K вопросов
- **Источник**: `squad_v2`
- **Описание**: Вопросы по текстам Wikipedia
- **Качество**: Высокое, созданы людьми
- **Загрузка**: `datasets.load_dataset('squad_v2')`

### Quora Question Pairs

- **Размер**: ~400K пар вопросов
- **Источник**: `quora`
- **Описание**: Пары похожих/разных вопросов с Quora
- **Качество**: Хорошее, краудсорсинг
- **Загрузка**: `datasets.load_dataset('quora')`

## 2. Специализированные NLP датасеты

### SNLI (Stanford Natural Language Inference)

- **Размер**: ~570K пар premise-hypothesis
- **Источник**: `snli`
- **Описание**: Логический вывод на естественном языке
- **Качество**: Высокое
- **Загрузка**: `datasets.load_dataset('snli')`

### MultiNLI

- **Размер**: ~433K пар
- **Источник**: `multi_nli`
- **Описание**: Многожанровый датасет для логического вывода
- **Качество**: Высокое
- **Загрузка**: `datasets.load_dataset('multi_nli')`

## 3. Диалоговые датасеты

### PersonaChat

- **Размер**: ~164K диалогов
- **Источник**: `facebook/persona_chat`
- **Описание**: Диалоги с заданными персонажами
- **Качество**: Хорошее
- **Загрузка**: `datasets.load_dataset('persona_chat')`

### Cornell Movie Dialogs

- **Размер**: ~220K диалогов
- **Источник**: Требует ручной загрузки
- **Описание**: Диалоги из фильмов
- **Качество**: Среднее
- **Ссылка**: https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html

## 4. Общие text-to-text датасеты

### OpenWebText

- **Размер**: ~8M документов
- **Источник**: `openwebtext`
- **Описание**: Веб-тексты, аналог GPT-2 training data
- **Качество**: Среднее-высокое
- **Загрузка**: `datasets.load_dataset('openwebtext')`

### Common Crawl

- **Размер**: Терабайты
- **Источник**: commoncrawl.org
- **Описание**: Архив интернета
- **Качество**: Смешанное
- **Примечание**: Требует значительной предобработки

## 5. Научные статьи

### ArXiv Dataset

- **Размер**: ~1.7M статей
- **Источник**: `arxiv_dataset`
- **Описание**: Научные статьи с arXiv.org
- **Качество**: Высокое
- **Загрузка**: `datasets.load_dataset('arxiv_dataset')`

### PubMed Central

- **Размер**: ~6M статей
- **Источник**: Требует API ключ
- **Описание**: Медицинские/биологические статьи
- **Качество**: Очень высокое

## 6. Рекомендации по выбору

### Для начального тестирования:

1. **SQuAD 2.0** - качественные вопросы-ответы, ~150K
2. **Quora Question Pairs** - разнообразные вопросы, ~400K
3. **SNLI** - логические связи, ~570K

### Для серьезного обучения:

1. **MS Marco** - самый большой качественный Q&A датасет
2. **Natural Questions** - реальные поисковые запросы
3. **OpenWebText** - разнообразный текстовый контент

### Для специализированных задач:

- **PersonaChat** - если нужны диалоги
- **ArXiv** - если нужна научная терминология
- **MultiNLI** - если важен логический вывод

## 7. Скрипт для загрузки популярных датасетов

```python
from datasets import load_dataset

# Загрузка разных датасетов
def download_dataset(name, size_limit=None):
    if name == "squad":
        dataset = load_dataset('squad_v2')
    elif name == "quora":
        dataset = load_dataset('quora')
    elif name == "snli":
        dataset = load_dataset('snli')
    elif name == "ms_marco":
        dataset = load_dataset('ms_marco', 'v1.1')
    elif name == "natural_questions":
        dataset = load_dataset('natural_questions')

    # Ограничение размера для тестирования
    if size_limit:
        dataset['train'] = dataset['train'].select(range(size_limit))

    return dataset
```

## 8. Преобразование в формат эмбеддингов

После загрузки любого из этих датасетов, можно адаптировать наш `generate_large_embedding_dataset.py` для работы с ними, заменив генерацию на загрузку и обработку готовых данных.

## Примечания:

- Некоторые датасеты требуют согласия с лицензией
- Большие датасеты могут занимать десятки GB
- Рекомендуется начать с меньших датасетов для тестирования
