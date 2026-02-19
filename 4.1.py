import numpy as np
import matplotlib.pyplot as plt

# ==============================
# 4.1 Визначення обсягу можливого фінансування
# ==============================

# Масив значень фінансування
x_vals = np.linspace(0, 2500, 1500)

# ------------------------------
# Функція для трапецієподібної належності
# ------------------------------
def trap_mu(x, left_peak, right_peak, alpha, beta, h):
    """
    left_peak, right_peak — плато
    alpha — лівий нахил
    beta — правий спад
    h — максимальна належність
    """
    mu = np.zeros_like(x, dtype=float)
    left = left_peak - alpha
    right = right_peak + beta

    # Лівий нахил
    if alpha > 0:
        idx = (x >= left) & (x < left_peak)
        mu[idx] = h * (x[idx] - left) / alpha

    # Плато
    mu[(x >= left_peak) & (x <= right_peak)] = h

    # Правий спад
    if beta > 0:
        idx = (x > right_peak) & (x <= right)
        mu[idx] = h * (right - x[idx]) / beta

    return mu

# ------------------------------
# Дані фінансових установ
# ------------------------------
A = {"left_peak": 300, "right_peak": 300, "alpha": 0, "beta": 0, "h": 1}
B = {"left_peak": 300, "right_peak": 350, "alpha": 50, "beta": 50, "h": 1}
C = {"left_peak": 200, "right_peak": 300, "alpha": 50, "beta": 50, "h": 1}
D = {"left_peak": 2100, "right_peak": 2200, "alpha": 100, "beta": 200, "h": 1}
E1 = {"left_peak": 300, "right_peak": 300, "alpha": 0, "beta": 200, "h": 0.2}
E2 = {"left_peak": 0, "right_peak": 0, "alpha": 0, "beta": 0, "h": 0.8}

# ------------------------------
# Побудова графіків окремо для кожної установи
# ------------------------------
def plot_fuzzy(x, mu, label, color):
    plt.figure(figsize=(7,4))
    plt.plot(x, mu, color=color)
    plt.title(f"Функція належності {label}")
    plt.xlabel("Сума фінансування, у.о.")
    plt.ylabel("μ(x)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

mu_A = trap_mu(x_vals, **A)
plot_fuzzy(x_vals, mu_A, "Установа A", "black")

mu_B = trap_mu(x_vals, **B)
plot_fuzzy(x_vals, mu_B, "Установа B", "blue")

mu_C = trap_mu(x_vals, **C)
plot_fuzzy(x_vals, mu_C, "Установа C", "green")

mu_D = trap_mu(x_vals, **D)
plot_fuzzy(x_vals, mu_D, "Установа D", "orange")

mu_E1 = trap_mu(x_vals, **E1)
plot_fuzzy(x_vals, mu_E1, "Установа E", "red")
# ------------------------------
# Об'єднання фінансування (ВИПРАВЛЕНО)
# ------------------------------
def sum_fuzzy(*args):
    return {
        "left_peak": sum(f["left_peak"] for f in args),
        "right_peak": sum(f["right_peak"] for f in args),
        "alpha": sum(f["alpha"] for f in args),
        "beta": sum(f["beta"] for f in args),
        "h": min(f["h"] for f in args)
    }

Base = sum_fuzzy(A, B, C, D)
S_with_E = sum_fuzzy(Base, E1)
S_without_E = sum_fuzzy(Base, E2)

# ------------------------------
# Вивід результатів (ВИПРАВЛЕНО КЛЮЧІ)
# ------------------------------
print("--- АНАЛІЗ ФІНАНСУВАННЯ ---")
print(f"Консервативний прогноз (без участі E, впевненість {S_without_E['h']}):")
print(f"  Найбільш ймовірна сума: {S_without_E['left_peak']} - {S_without_E['right_peak']} у.о.")
print(f"  Повний можливий діапазон: {S_without_E['left_peak'] - S_without_E['alpha']} - {S_without_E['right_peak'] + S_without_E['beta']} у.о.\n")

print(f"Оптимістичний прогноз (з участю E, впевненість {S_with_E['h']}):")
print(f"  Найбільш ймовірна сума: {S_with_E['left_peak']} - {S_with_E['right_peak']} у.о.")
print(f"  Повний можливий діапазон: {S_with_E['left_peak'] - S_with_E['alpha']} - {S_with_E['right_peak'] + S_with_E['beta']} у.о.\n")

print("Висновок:")
print(f"  Найбільш надійна сума фінансування: [{S_without_E['left_peak']}, {S_without_E['right_peak']}] у.о.")
print(f"  Додаткове залучення E дозволяє збільшити бюджет до [{S_with_E['left_peak']}, {S_with_E['right_peak']}] у.о.,")
print(f"  але впевненість у такій сумі знижується до {S_with_E['h']}.")
# ------------------------------
# Спільний графік
# ------------------------------
x_sum = np.linspace(2000, 4500, 2000)
mu_S1 = trap_mu(x_sum, **S_with_E)
mu_S2 = trap_mu(x_sum, **S_without_E)
mu_total = np.maximum(mu_S1, mu_S2)

plt.figure(figsize=(8,4))
plt.plot(x_sum, mu_S2, '--', color='blue', label=f"S без E (h={S_without_E['h']})")
plt.plot(x_sum, mu_S1, '--', color='orange', label=f"S з E (h={S_with_E['h']})")
plt.plot(x_sum, mu_total, color='black', linewidth=2, label="Результуюча μ(x)")
plt.xlabel("Загальна сума фінансування, у.о.")
plt.ylabel("μ(x)")
plt.title("Об'єднана функція належності фінансування")
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend()
plt.show()
