#!/usr/bin/env python3
"""
Ультра-минимальный анализ gMLP - где именно расходуются параметры?
"""


def analyze_parameter_breakdown():
    """Детальный анализ того, где расходуются параметры"""

    print("=== АНАЛИЗ РАСХОДА ПАРАМЕТРОВ В МИНИМАЛЬНОЙ gMLP ===")
    print()

    # Текущие размеры из логов
    state_size = 4
    neighbor_count = 6
    hidden_dim = 3  # МИНИМУМ
    external_input_size = 1  # МИНИМУМ

    # Вычисляем входной размер
    neighbor_input_size = neighbor_count * state_size  # 6 * 8 = 48
    total_input_size = (
        neighbor_input_size + state_size + external_input_size
    )  # 48 + 8 + 1 = 57

    print(f"ВХОДНЫЕ РАЗМЕРЫ:")
    print(f"  state_size: {state_size}")
    print(f"  neighbor_count: {neighbor_count}")
    print(f"  neighbor_input_size: {neighbor_input_size}")
    print(f"  external_input_size: {external_input_size}")
    print(f"  total_input_size: {total_input_size}")
    print()

    print("ПОЭТАПНЫЙ РАСЧЕТ ПАРАМЕТРОВ:")

    # 1. Input LayerNorm
    input_norm_params = total_input_size * 2  # weight + bias
    print(f"1. Input LayerNorm: {input_norm_params} params")
    print(f"   Формула: {total_input_size} * 2 = {input_norm_params}")

    # 2. Input Projection
    input_proj_params = total_input_size * hidden_dim + hidden_dim  # W + b
    print(f"2. Input Projection: {input_proj_params} params")
    print(
        f"   Формула: {total_input_size} * {hidden_dim} + {hidden_dim} = {input_proj_params}"
    )

    # 3. Output LayerNorm
    output_norm_params = hidden_dim * 2
    print(f"3. Output LayerNorm: {output_norm_params} params")
    print(f"   Формула: {hidden_dim} * 2 = {output_norm_params}")

    # 4. Output Projection
    output_proj_params = hidden_dim * state_size + state_size
    print(f"4. Output Projection: {output_proj_params} params")
    print(
        f"   Формула: {hidden_dim} * {state_size} + {state_size} = {output_proj_params}"
    )

    # 5. Residual connection (если нужен)
    if total_input_size != state_size:
        residual_params = total_input_size * state_size + state_size
        print(f"5. Input Residual: {residual_params} params")
        print(
            f"   Формула: {total_input_size} * {state_size} + {state_size} = {residual_params}"
        )
    else:
        residual_params = 0
        print(f"5. Input Residual: {residual_params} params (не нужен)")

    total_params = (
        input_norm_params
        + input_proj_params
        + output_norm_params
        + output_proj_params
        + residual_params
    )

    print()
    print(f"ИТОГО: {total_params} параметров")
    print(f"TARGET: 300 параметров")
    print(f"ПРЕВЫШЕНИЕ: {total_params - 300} параметров ({total_params/300:.1f}x)")
    print()

    # Анализ основного "пожирателя" параметров
    print("ОСНОВНЫЕ 'ПОЖИРАТЕЛИ' ПАРАМЕТРОВ:")
    components = [
        ("Input LayerNorm", input_norm_params),
        ("Input Projection", input_proj_params),
        ("Output LayerNorm", output_norm_params),
        ("Output Projection", output_proj_params),
        ("Input Residual", residual_params),
    ]

    for name, params in sorted(components, key=lambda x: x[1], reverse=True):
        percent = params / total_params * 100
        print(f"  {name}: {params} params ({percent:.1f}%)")

    return total_params


def find_ultra_minimal_config():
    """Ищем ультра-минимальную конфигурацию"""

    print()
    print("=== ПОИСК УЛЬТРА-МИНИМАЛЬНОЙ КОНФИГУРАЦИИ ===")
    print()

    target = 300
    best_configs = []

    # Пробуем более радикальные уменьшения
    for state_size in [4, 6, 8]:  # Уменьшаем state_size
        for neighbor_count in [4, 6]:  # Пробуем 4-связность вместо 6
            for hidden_dim in [1, 2, 3]:
                for external_input_size in [0, 1]:  # Даже без external input

                    neighbor_input_size = neighbor_count * state_size
                    total_input_size = (
                        neighbor_input_size + state_size + external_input_size
                    )

                    # Минимальная архитектура: только input_proj + output_proj
                    params = 0
                    params += total_input_size * 2  # input norm
                    params += total_input_size * hidden_dim + hidden_dim  # input proj
                    params += hidden_dim * 2  # output norm
                    params += hidden_dim * state_size + state_size  # output proj

                    # Residual если нужен
                    if total_input_size != state_size:
                        params += total_input_size * state_size + state_size

                    diff = abs(params - target)

                    config = {
                        "state_size": state_size,
                        "neighbor_count": neighbor_count,
                        "hidden_dim": hidden_dim,
                        "external_input_size": external_input_size,
                        "total_input_size": total_input_size,
                        "params": params,
                        "diff": diff,
                        "ratio": params / target,
                    }

                    best_configs.append(config)

    # Сортируем по близости к target
    best_configs.sort(key=lambda x: x["diff"])

    print("ТОП-5 БЛИЖАЙШИХ К TARGET КОНФИГУРАЦИЙ:")
    for i, config in enumerate(best_configs[:5]):
        print(
            f"{i+1}. {config['params']} params (target: {target}, diff: {config['diff']:+d})"
        )
        print(
            f"   state_size={config['state_size']}, neighbor_count={config['neighbor_count']}"
        )
        print(
            f"   hidden_dim={config['hidden_dim']}, external_input={config['external_input_size']}"
        )
        print(f"   ratio: {config['ratio']:.2f}x")
        print()

    return best_configs[0]


def suggest_architecture_changes():
    """Предлагает изменения архитектуры для достижения target"""

    print("=== ПРЕДЛОЖЕНИЯ ПО АРХИТЕКТУРЕ ===")
    print()

    print("🎯 ПРОБЛЕМА: Даже минимальная конфигурация слишком тяжелая")
    print()

    print("💡 ВОЗМОЖНЫЕ РЕШЕНИЯ:")
    print()

    print("1. 📐 УМЕНЬШИТЬ БАЗОВЫЕ РАЗМЕРЫ:")
    print("   • state_size: 8 → 4-6 (уменьшить состояние клетки)")
    print("   • neighbor_count: 6 → 4 (4-связность вместо 6-связности)")
    print("   • external_input_size: 1 → 0 (убрать внешний вход совсем)")
    print()

    print("2. 🔧 УПРОСТИТЬ АРХИТЕКТУРУ:")
    print("   • Убрать LayerNorm слои (экономия ~100-200 параметров)")
    print("   • Убрать bias в Linear слоях")
    print("   • Объединить input_projection + output_projection в один слой")
    print()

    print("3. 🧠 АЛЬТЕРНАТИВНАЯ АРХИТЕКТУРА:")
    print("   • Простая MLP без gating механизмов")
    print("   • Прямое линейное преобразование neighbor → own_state")
    print("   • Добавить нелинейность только в критических местах")
    print()

    print("4. 🔄 ПЕРЕОСМЫСЛИТЬ ПОДХОД:")
    print("   • Возможно, 300 параметров слишком мало для gMLP?")
    print("   • Рассмотреть target 500-800 параметров как компромисс?")
    print("   • Или создать совсем другую архитектуру (не gMLP)")


def main():
    total_params = analyze_parameter_breakdown()
    best_config = find_ultra_minimal_config()
    suggest_architecture_changes()

    print()
    print("=== ВЫВОДЫ ===")
    print(f"• Текущая минимальная gMLP: {total_params} параметров")
    print(f"• Лучшая найденная конфигурация: {best_config['params']} параметров")
    print(f"• Для target=300 нужны радикальные изменения архитектуры")
    print(f"• Рекомендация: пересмотреть target или архитектуру")


if __name__ == "__main__":
    main()
