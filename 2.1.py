import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === Вибір логічної функції (варіант 1) ===
def logic_variant_1(x):
    x1, x2, x3 = x
    return int(x1 or (x2 and x3))

# === Вхідні дані ===
X = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
])

T = np.array([logic_variant_1(x) for x in X])

# === Перцептрон з можливістю зміщення та порогу ===
class Perceptron:
    def __init__(self, eta=0.2, theta=0.0, max_epochs=1000):
        self.eta = eta      # коефіцієнт навчання
        self.theta = theta  # порогове значення
        self.max_epochs = max_epochs
        self.w = np.random.uniform(-0.5, 0.5, X.shape[1])
        self.b = np.random.uniform(-0.5, 0.5)  # зміщення

    def activation(self, x):
        return np.dot(self.w, x) + self.b

    def output(self, x):
        return int(self.activation(x) >= self.theta)

    def train_epoch(self, X, T):
        rows = []
        errors = 0
        for x, t in zip(X, T):
            a = self.activation(x)
            y = self.output(x)
            delta = t - y

            # Зберігаємо проміжні значення для звіту
            rows.append([
                *self.w, self.b, self.theta,
                *x, a, y, t,
                self.eta * delta, delta
            ])

            # Оновлення ваг
            if delta != 0:
                self.w += self.eta * delta * x
                self.b += self.eta * delta
                errors += 1
        return rows, errors

    def train(self, X, T):
        for epoch in range(self.max_epochs):
            _, errors = self.train_epoch(X, T)
            if errors == 0:
                return True, epoch + 1  # збіжність досягнута
        return False, self.max_epochs  # збіжність не досягнута

# === Тестування для різних коефіцієнтів навчання ===
etas = [0.2, 0.4, 0.6]
epochs_needed = []

for eta in etas:
    p = Perceptron(eta=eta, theta=0.0)
    converged, epochs = p.train(X, T)
    epochs_needed.append(epochs)
    print(f"eta={eta}: {'Збіжність' if converged else 'Не збігається'} за {epochs} епох")

# === Графік швидкості збіжності ===
plt.figure()
plt.plot(etas, epochs_needed, marker='o')
plt.xlabel('Коефіцієнт навчання η')
plt.ylabel('Кількість епох до збіжності')
plt.title('Швидкість збіжності перцептрона (варіант 1)')
plt.grid(True)
plt.show()

# === Звітна таблиця для останнього запуску ===
p = Perceptron(eta=0.2, theta=0.0)
rows, _ = p.train_epoch(X, T)

columns = [
    "w1", "w2", "w3", "b", "θ",
    "x1", "x2", "x3",
    "a", "Y", "T", "η·(T−Y)", "δ"
]

df = pd.DataFrame(rows, columns=columns)
print(df)

# === Перевірка на лінійну подільність для XOR ===
def xor_function(x):
    x1, x2, _ = x
    return int(x1 != x2)

T_xor = np.array([xor_function(x) for x in X])
p_xor = Perceptron(eta=0.2, theta=0.0)
converged_xor, _ = p_xor.train(X, T_xor)
print("XOR:", "лінійно подільна" if converged_xor else "лінійно НЕ подільна")
