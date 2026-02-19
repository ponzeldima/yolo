import numpy as np
import matplotlib.pyplot as plt

# ======================
# ВИХІДНІ ДАНІ (табл. 1.8)
# ======================
x1 = np.array([1, 5, 12, 23, 34, 53, 66, 69, 78], dtype=float)
x2 = np.array([88, 77, 66, 56, 43, 34, 31, 23, 22], dtype=float)
x3 = np.array([11, 32, 34, 45, 48, 65, 77, 88, 96], dtype=float)
y  = np.array([2, 4, 8, 12, 17, 32, 54, 65, 77], dtype=float)

N = len(y)

# ======================
# ФОРМУВАННЯ МАТРИЦІ МОДЕЛІ
# ======================
X = np.column_stack((
    np.ones(N),
    x1,
    x2,
    x3
))

# ======================
# ОЦІНКА ПАРАМЕТРІВ МНК
# ======================
beta = np.linalg.inv(X.T @ X) @ X.T @ y
b0, b1, b2, b3 = beta

print("Оцінені коефіцієнти множинної регресії:")
print(f"b0 = {b0:.4f}")
print(f"b1 = {b1:.4f}")
print(f"b2 = {b2:.4f}")
print(f"b3 = {b3:.4f}")

print(f"\nРівняння регресії:")
print(f"Y = {b0:.4f} + {b1:.4f}·X1 + {b2:.4f}·X2 + {b3:.4f}·X3")

# ======================
# ПРОГНОЗУВАННЯ
# ======================
x_new = np.array([1, 33, 50, 54])   # [1, X1, X2, X3]
y_forecast = x_new @ beta
print(f"\nПрогнозоване значення Y: {y_forecast:.2f}")

# ======================
# ПЕРЕВІРКА АДЕКВАТНОСТІ
# ======================
y_calc = X @ beta
errors = y - y_calc

SSE = np.sum(errors**2)
SST = np.sum((y - np.mean(y))**2)
R_squared = 1 - SSE / SST

print(f"\nКоефіцієнт детермінації R² = {R_squared:.4f}")

if R_squared >= 0.9:
    print("Висновок: модель має високу адекватність (можлива мультиколінеарність).")
elif R_squared >= 0.7:
    print("Висновок: модель є адекватною.")
else:
    print("Висновок: модель недостатньо адекватна.")

# ======================
# КОЕФІЦІЄНТИ ЕЛАСТИЧНОСТІ
# ======================
Ex1 = b1 * np.mean(x1) / np.mean(y)
Ex2 = b2 * np.mean(x2) / np.mean(y)
Ex3 = b3 * np.mean(x3) / np.mean(y)

print("\nКоефіцієнти еластичності:")
print(f"E1 = {Ex1:.3f}")
print(f"E2 = {Ex2:.3f}")
print(f"E3 = {Ex3:.3f}")

# ======================
# ГРАФІЧНИЙ АНАЛІЗ
# ======================
plt.figure(figsize=(7, 4))
plt.plot(y, marker='o', label="Фактичні значення")
plt.plot(y_calc, linestyle='--', label="Розрахункові значення")
plt.xlabel("Номер спостереження")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(7, 4))
plt.stem(errors)
plt.title("Залишки регресійної моделі")
plt.xlabel("Номер спостереження")
plt.ylabel("Похибка")
plt.grid(True)
plt.show()
