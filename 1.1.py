import matplotlib.pyplot as plt

# Вихідні статистичні дані (табл. 1.7)
x_vals = [24.32, 28.34, 34.56, 39.45, 44.76,
          50.32, 55.34, 60.43, 65.87, 88.98]

y_vals = [76.33, 70.34, 65.82, 60.23, 54.99,
          50.22, 45.74, 40.34, 34.84, 30.23]

m = len(x_vals)

# Обчислення середніх значень
x_avg = sum(x_vals) / m
y_avg = sum(y_vals) / m

# Допоміжні суми для МНК
Sxy = sum((x_vals[i] - x_avg) * (y_vals[i] - y_avg) for i in range(m))
Sxx = sum((x - x_avg) ** 2 for x in x_vals)

# Оцінка параметрів лінійної регресії
beta_1 = Sxy / Sxx          # коефіцієнт нахилу
beta_0 = y_avg - beta_1 * x_avg  # вільний член

print(f"Оцінка коефіцієнта β₁ = {beta_1:.4f}")
print(f"Оцінка коефіцієнта β₀ = {beta_0:.4f}")

# Прогнозування значення Y при X = 88.98
x_forecast = 88.98
y_forecast = beta_0 + beta_1 * x_forecast
print(f"Прогнозоване значення Y при X = {x_forecast}: {y_forecast:.2f}")

# Побудова лінії регресії
y_model = [beta_0 + beta_1 * x for x in x_vals]

plt.figure(figsize=(7, 5))
plt.scatter(x_vals, y_vals, marker='o', label="Емпіричні дані")
plt.plot(x_vals, y_model, linestyle='-', label="Лінійна регресія")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Парна лінійна регресія")
plt.legend()
plt.grid(True)
plt.show()
