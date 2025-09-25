тобы подготовить детальный план Фазы 1, я:
• просмотрел docs/textcube_tokenizer_plan.md и исходники в EnergyFlowSwift/Sources/EnergyFlow/TextBridge
• проинвентаризировал текущую реализацию и выявил, что можно переиспользовать

Краткий анализ текущего состояния
• Уже есть:
◦ TextBridge API-слой для векторизации: TextToCubeEncoder, TextToCubeEncoderConfig
◦ Простейший токенизатор SimpleTokenizer, дающий ids и attentionMask (разделение по пробелам, динамический словарь)
◦ Пайплайн: ids -> Embedding -> (Transformer encoder, masked mean pool) -> MLP -> выход
• Несоответствие с новой идеей (Text→Cube Tokenizer):
◦ Нужен детерминированный токенизатор с формальной грамматикой, header/body, фиксированными ID, обратимостью и mask, защищающей header/CRC
◦ TextBridge должен быть «тонким мостом» (без Transformer/MLP). Интеллект — в Cube
◦ Требуется 1D→2D (Hilbert) и MPSGraph-embedding/reshape (позже, не в Фазе 1)
• Что можно переиспользовать в дальнейшем:
◦ Паттерны работы с масками и паддингом из TextToCubeEncoder.padOrTruncate
◦ Логгер и общий фрейм для отладки/интеграции
◦ Параметризацию размеров/длин из TextToCubeEncoderConfig (вместо maxLength использовать L=256 для surface 16×16)
предполагается, помимо прочего, что будет текстовое поясниние сложной логики, общая задача и задача на текущую логическую итерацию... но это уже другой вопрос по реализации датасета и логических потоков. но где-то нужно будет подумать о том, что могут быть выделенные места под утвержден, общуб цель и локальную цель и общий формат может быть вида <global_goal>, <local_goal>, <stmt> или как-то так на каком-то этапе это можно будет добавить, когда база будет работать.

Ниже — подробный .md план для Фазы 1.

План Фазы 1: Text→Cube Tokenizer (MVP-ядро) для EnergyFlowSwift

1. Цели и ограничения фазы
   • Реализовать обратимый токенизатор для логических блоков по MVP-грамматике из плана:
   ◦ Лексер/парсер, SymbolTable, header/body, фиксированные ID
   ◦ Round-trip: decode(encode(x)) == x для длины ≤ 256 токенов
   • Сгенерировать mask для защищённых участков (только header в Фазе 1; CRC отложим)
   • Интеграция не ломающая текущий код:
   ◦ Сосуществование с SimpleTokenizer
   ◦ Новый API и типы находятся рядом, но не подключают MPS/Hilbert/CRC/Transformer
   • Не делаем в Фазе 1:
   ◦ CRC секцию
   ◦ Hilbert 1D↔2D
   ◦ MPSGraph embedder/reshape
   ◦ Удаление/рефакторинг TextToCubeEncoder

2. Структура кода (раскладка по файлам)
   • Новые файлы в EnergyFlowSwift/Sources/EnergyFlow/TextBridge:
   ◦ TokenSequence.swift — публичная структура токен-последовательности и mask
   ◦ TextTokenizer.swift — публичный протокол TextTokenizer
   ◦ Vocab.swift — enum/структура с фиксированными ID токенов (из §3 плана, MVP-подмножество)
   ◦ TokenizerCore/
   ▪ Lexer.swift — детерминированный лексер под BNF из §2
   ▪ Parser.swift — простой парсер в AST (stmt/term/rel/number/ident)
   ▪ SymbolTable.swift — Vi↔имя, первый проход (V0..V63)
   ◦ Encoder/
   ▪ TextCubeTokenizer.swift — реализация TextTokenizer.encode
   ▪ HeaderBuilder.swift — секция словаря имён (HDR_BOS..HDR_END)
   ▪ BodyBuilder.swift — секция тела (BODY_BOS..BODY_END), числа и индексы
   ◦ Decoder/
   ▪ TextCubeDetokenizer.swift — decode для round-trip
   • Тесты (EnergyFlowSwift/Tests/EnergyFlowTests/TextBridge):
   ◦ TextCubeTokenizerTests.swift — набор эталонов и property-тест round-trip

3. Скоуп функциональности (MVP-грамматика)
   • Поддерживаем:
   ◦ Блоки вида: (x<y, y<z), (a<=b, b<=c, a<=c), (u=v, v=w, u=w)
   ◦ Отношения: <, <=, =, !=, >, >=
   ◦ Идентификаторы: [A-Za-z\_][A-Za-z0-9_]_
   ◦ Числа: [+-]?[0-9]+ (как последовательности NUM*BOS, SIGN?, DIGIT*_, NUM_END)
   • Header:
   ◦ MAP_PAIR с IDX_BOS/IDX_END и BYTES_BOS/END для UTF-8 имён
   ◦ Vi ограничено V0..V63
   • Body и EOS/PAD:
   ◦ BODY_BOS..BODY_END, затем EOS; паддинг до 256 PAD-ами
   • Mask:
   ◦ mask=1 для диапазона [HDR_BOS..HDR_END]
   ◦ mask=0 для тела и PAD (в Фазе 1 без CRC)
   • Валидация входа:
   ◦ Длина ≤ 256 после кодирования (иначе ошибка для MVP)

4. Публичный API
   • Типы:
   ◦ TokenSequence: ids [Int32] длиной ровно 256, mask [UInt8] длиной 256, len Int (фактическая длина до PAD)
   • Протокол:
   ◦ TextTokenizer: encode(:) -> TokenSequence; decode(:) -> String
   • Имплементация:
   ◦ TextCubeTokenizer: init(options?) с фиксированным словарём ID (Vocab)
   • Адаптеры (без ломки существующего кода):
   ◦ В отдельном утилити добавить преобразование TokenSequence -> EncodedBatch для совместимости с текущими методами, если понадобится: ids как [[Int]] с B=1, attentionMask из mask или из позиций непустых токенов тела. На Фазе 1 использовать только в тестах или экспериментально, без включения в путь TextToCubeEncoder по умолчанию

5. Реализация: шаги и контрольные точки
   • Шаг 1. Vocab и ID-схема (фиксированная)
   ◦ Выписать подмножество ID из §3 (PAD, BOS/EOS/SEP, HDR*, BODY*, скобки/знаки, NUM/IDX/байты, DIGIT*0..9)
   ◦ Описать в Vocab.swift как enum с rawValue Int и утилиты toID(*:), безопасные диапазоны для DIGIT/IDX/BYTE
   ◦ Критерий готовности: компилируется, покрыт sanity-тестами на диапазоны
   • Шаг 2. Lexer + Parser (минимум)
   ◦ Лексер для идентификаторов, чисел, операторов и разделителей
   ◦ Парсер для block -> список stmt, stmt -> term rel term | term -> term, без импликации на MVP, если не нужна с ходу
   ◦ Критерий: корректный AST для 10+ эталонов
   • Шаг 3. SymbolTable и header
   ◦ Строим Vi для имён по порядку появления
   ◦ Генерация header секции по схеме MAP*PAIR + IDX*\_ + BYTES\__ (UTF-8 байты имени)
   ◦ Критерий: header восстанавливает словарь имён в декодере
   • Шаг 4. Body и числа
   ◦ Генерация токенов тела с использованием IDX*BOS/IDX_END и операторов rel
   ◦ Числа: NUM_BOS/END + SIGN*_ + DIGIT\_\_
   ◦ Критерий: ids соответствуют ожидаемым паттернам на эталонах
   • Шаг 5. Длина, EOS и паддинг
   ◦ Добавить EOS, паддинг до 256, len вычислять до первого PAD
   ◦ Критерий: длины и паддинг стабильны, без выхода за 256
   • Шаг 6. Mask
   ◦ mask=1 на всём диапазоне header; остальное 0
   ◦ Критерий: проверяется в encode-тестах
   • Шаг 7. Decoder
   ◦ Парсинг header назад в таблицу имён, печать тела с подстановкой
   ◦ Критерий: decode(encode(x)) == x на эталонах
   • Шаг 8. Тесты и проверки
   ◦ Набор кейсов из плана (§8, §9), property-тест round-trip, негативные кейсы ошибки парсинга
   ◦ Критерии: все тесты проходят локально; покрытие ключевых веток

6. Интеграция и совместимость
   • Не трогать TextToCubeEncoder, TextToCubeEncoderConfig и SimpleTokenizer в Фазе 1
   • Добавить новый модуль к публичной сборке, но не включать в текущие производственные пути
   • Предусмотреть утилиту-конвертер TokenSequence -> EncodedBatch для экспериментов в тестах
   • Оставить заметку в README по TextBridge о том, что новый токенизатор — экспериментальный, с планом интеграции на последующие фазы

7. Критерии готовности Фазы 1
   • Публичные типы TokenSequence, TextTokenizer, TextCubeTokenizer доступны и документированы
   • Round-trip без потерь на L≤256 для набора эталонов
   • mask корректно помечает header
   • Полный набор юнит-тестов зелёный
   • Отсутствуют изменения в текущих продовых путях векторизации

8. Риски и упрощения
   • Ограничение Vi до 64 имён — принять в Фазе 1, расширение позже
   • Отсутствие CRC — осознанно, добавим в следующую фазу
   • Числа без десятичной точки и экспоненты в Фазе 1
   • Импликацию (->) можно отложить, если не нужна для первых сценариев

9. Следующие фазы (для ориентира)
   • Фаза 2: CRC и mask для CRC, Hilbert 1D↔2D, адаптер к surface 16×16, базовая интеграция с Cube
   • Фаза 3: MPSGraph embedder/reshape, бенчмарки, отказ от промежуточного Transformer/MLP в TextBridge
   • Фаза 4: расширение грамматики, IDX_EXT, генерация словаря из JSON/YAML, Interop
