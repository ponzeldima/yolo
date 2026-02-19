
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Визначення функцій належності (Membership Functions) ---
 
def mu_trapezoid(x, a, b, c, d):
    """Трапецієподібна функція належності"""
    y = np.zeros_like(x, dtype=float)
    # Зростання
    m1 = (x >= a) & (x < b)
    if b > a: y[m1] = (x[m1] - a) / (b - a)
    # Плато (ядро)
    m2 = (x >= b) & (x <= c)
    y[m2] = 1.0
    # Спадання
    m3 = (x > c) & (x <= d)
    if d > c: y[m3] = (d - x[m3]) / (d - c)
    return np.clip(y, 0, 1)

def mu_crisp(x, val, eps=0.5):
    """Точне значення (функція-пік)"""
    return (np.abs(x - val) < eps).astype(float)

def mu_rising(x, left, right):
    """Зростаючий сегмент (установа В)"""
    y = np.zeros_like(x, dtype=float)
    mask = (x >= left) & (x <= right)
    y[mask] = (x[mask] - left) / (right - left)
    return np.c~lip(y, 0, 1)

def mu_falling(x, left, right):
    """Спадний сегмент (установа Д)"""
    y = np.zeros_like(x, dtype=float)
    mask = (x >= left) & (x <= right)
    y[mask] = (right - x[mask]) / (right - left)
    return np.clip(y, 0, 1)

# --- 2. Допоміжна функція для виділення інтервалів ---

def get_intervals(x, mask):
    """Перетворює маску значень у список читабельних інтервалів"""
    if not np.any(mask): return []
    idx = np.where(mask)[0]
    intervals = []
    start = idx[0]
    for i in range(1, len(idx)):
        if idx[i] != idx[i-1] + 1:
            intervals.append((x[start], x[idx[i-1]]))
            start = idx[i]
    intervals.append((x[start], x[idx[-1]]))
    return intervals

# --- 3. Основний розрахунок (Агрегація) ---

# Діапазон значень фінансування і "точність" сітки.
# Важливо: щоб точкове значення 300 у.о. точно було присутнє в x,
# використовуємо рівномірну сітку з кроком dx (за замовчуванням 1 у.о.).
x_min, x_max = 0, 2600
dx = 1  # змініть, наприклад, на 0.5 або 0.1, якщо потрібна густіша сітка
x = np.arange(x_min, x_max + dx, dx, dtype=float)

# Опис установ за умовами задачі (функції належності μ(x)):
# 1) А: "точно 300 у.о."
mu_A = mu_crisp(x, 300, eps=max(dx / 2, 1e-9))
# 2) Б: підтримка 250..400, ядро 300..350
mu_B = mu_trapezoid(x, 250, 300, 350, 400)
# 3) В: 200..300 із зростанням упевненості при збільшенні суми
mu_V = mu_rising(x, 200, 300)
# 4) Г: 2000..2400, більш надійно 2100..2200
mu_G = mu_trapezoid(x, 2000, 2100, 2200, 2400)
# 5) Д: 300..500 із падінням упевненості при збільшенні суми
mu_D = mu_falling(x, 300, 500)

# Агрегація за принципом "АБО" (MAX): проект може фінансуватись будь-якою установою
mu_total = np.maximum.reduce([mu_A, mu_B, mu_V, mu_G, mu_D])

# Визначення характеристик
core_thr = 0.99   # поріг для ядра (μ ≈ 1)
zero_thr = 0.01   # поріг для неможливих значень (μ ≈ 0)
most_possible = get_intervals(x, mu_total >= core_thr)  # Ядро (найбільш можливі)
impossible = get_intervals(x, mu_total < zero_thr)      # Неможливі значення

# --- 4. Візуалізація результатів ---

plt.figure(figsize=(12, 6))
plt.plot(x, mu_total, label='Агрегована можливість (TOTAL)', color='grey', linewidth=3, alpha=0.7)
plt.fill_between(x, mu_total, color='gray', alpha=0.2)
plt.plot(x, mu_A, '--', label='А (300)', alpha=0.6)
plt.plot(x, mu_B, '--', label='Б (трапеція)', alpha=0.6)
plt.plot(x, mu_V, '--', label='В (200→300, зростання)', alpha=0.6)
plt.plot(x, mu_G, '--', label='Г (велика сума)', alpha=0.6)
plt.plot(x, mu_D, '--', label='Д (300→500, спадання)', alpha=0.6)

plt.title('Аналіз можливого фінансування інвестиційного проекту')
plt.xlabel('Сума фінансування (у.о.)')
plt.ylabel('Ступінь впевненості (μ)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# --- 5. Вивід результатів ---

print("РЕЗУЛЬТАТИ АНАЛІЗУ:")
print(f"Найбільш можливі суми фінансування (ядро): {most_possible}")
print(f"Неможливі обсяги (в межах {x_min}-{x_max}): {impossible}")