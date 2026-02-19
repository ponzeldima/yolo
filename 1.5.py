# ===============================
# 1.5  МЕТОД БРАНДОНА (нелінійна регресія)
# ===============================

import numpy as np

print("\n1.5 МЕТОД БРАНДОНА")

# Вихідні дані (табл. 1.8)
X1 = np.array([1, 5, 12, 23, 34, 53, 66, 69, 78], dtype=float)
X2 = np.array([88, 77, 66, 56, 43, 34, 31, 23, 22], dtype=float)
X3 = np.array([11, 32, 34, 45, 48, 65, 77, 88, 96], dtype=float)
Y  = np.array([2, 4, 8, 12, 17, 32, 54, 65, 77], dtype=float)

n = len(Y)

# Логарифмування змінних
Y_log  = np.log(Y)
X1_log = np.log(X1)
X2_log = np.log(X2)
X3_log = np.log(X3)

# Формування матриці регресорів
Zb = np.column_stack([
    np.ones(n),
    X1_log,
    X2_log,
    X3_log
])

# Оцінювання параметрів МНК
theta = np.linalg.inv(Zb.T @ Zb) @ (Zb.T @ Y_log)

ln_a, b1, b2, b3 = theta
a = np.exp(ln_a)

print("\nОцінені параметри моделі:")
print("a =", a)
print("b1 =", b1)
print("b2 =", b2)
print("b3 =", b3)

# Аналітичний вигляд моделі
print("\nРівняння нелінійної регресії (метод Брандона):")
print(f"Y = {a:.4f} · X1^{b1:.4f} · X2^{b2:.4f} · X3^{b3:.4f}")

# Теоретичні значення
Y_log_hat = Zb @ theta
Y_hat = np.exp(Y_log_hat)

# Оцінка адекватності
SSE = np.sum((Y - Y_hat)**2)
SST = np.sum((Y - np.mean(Y))**2)
R2 = 1 - SSE / SST

print("\nОцінка адекватності моделі:")
print("R^2 =", R2)

if R2 >= 0.9:
    print("Висновок: модель має дуже високу адекватність.")
elif R2 >= 0.7:
    print("Висновок: модель є адекватною.")
else:
    print("Висновок: модель недостатньо адекватна.")

# Прогнозування
X1_new, X2_new, X3_new = 33, 50, 54
Y_forecast = a * (X1_new**b1) * (X2_new**b2) * (X3_new**b3)

print("\nПрогноз при X1=33, X2=50, X3=54:")
print("Y =", Y_forecast)

# Інтерпретація (еластичності)
print("\nІнтерпретація коефіцієнтів:")
print(f"b1 = {b1:.4f} → +1% X1 ⇒ {b1:.4f}% Y")
print(f"b2 = {b2:.4f} → +1% X2 ⇒ {b2:.4f}% Y")
print(f"b3 = {b3:.4f} → +1% X3 ⇒ {b3:.4f}% Y")
