import numpy as np
import matplotlib.pyplot as plt

# ==============================
# Параметри трапецієподібних множин: (m1, m2, alpha, beta, h)
# ==============================
A1 = (100, 200, 30, 40, 1)
A2 = (200, 300, 20, 60, 1)
B1 = (140, 240, 30, 40, 1)
B2 = (240, 320, 50, 40, 1)
C1 = (50, 100, 10, 30, 1)
C2 = (100, 150, 20, 50, 1)

x0, y0 = 220, 200

# ==============================
# Трапецієподібна функція належності
# ==============================
def trap(x, m1, m2, alpha, beta, h):
    if x < m1 - alpha or x > m2 + beta:
        return 0.0
    elif m1 <= x <= m2:
        return h
    elif m1 - alpha <= x < m1:
        return h * (x - (m1 - alpha)) / alpha
    elif m2 < x <= m2 + beta:
        return h * ((m2 + beta) - x) / beta
    return 0.0

# ==============================
# 1. Фаззифікація вхідних змінних
# ==============================
muA1 = trap(x0, *A1)
muA2 = trap(x0, *A2)
muB1 = trap(y0, *B1)
muB2 = trap(y0, *B2)

# ==============================
# 2. Активація правил (AND = min)
# ==============================
alpha1 = min(muA1, muB1)
alpha2 = min(muA2, muB2)

print(f"Рівні активації правил:")
print(f"Правило P1: α1 = {alpha1:.3f}")
print(f"Правило P2: α2 = {alpha2:.3f}")

# ==============================
# 3. Активація висновків
# ==============================
z_values = np.linspace(0, 250, 2000)

def activate_rule(z, params, alpha):
    return np.array([min(alpha, trap(v, *params)) for v in z])

muC1 = activate_rule(z_values, C1, alpha1)
muC2 = activate_rule(z_values, C2, alpha2)

# Агрегація (OR = max)
muZ = np.maximum(muC1, muC2)

# ==============================
# 4. Дефазифікація (центр ваги)
# ==============================
if np.trapz(muZ, z_values) == 0:
    Z0 = 0.0
else:
    Z0 = np.trapz(z_values * muZ, z_values) / np.trapz(muZ, z_values)

print(f"Результат дефазифікації: Z0 = {Z0:.2f}")

# ==============================
# 5. Візуалізація фаззифікації та висновків
# ==============================
plt.figure(figsize=(10, 4))
plt.plot(z_values, muC1, color='blue', alpha=0.6, label=f'C1 активована (α1={alpha1:.2f})')
plt.plot(z_values, muC2, color='orange', alpha=0.6, label=f'C2 активована (α2={alpha2:.2f})')
plt.fill_between(z_values, muZ, color='green', alpha=0.3, label='Об’єднана μZ(z)')
plt.axvline(Z0, color='red', linestyle='--', linewidth=2, label=f'Z0 = {Z0:.2f}')
plt.xlabel('Z')
plt.ylabel('μ(Z)')
plt.title('Нечітке логічне виведення (метод Мамдані)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right')
plt.show()
