import numpy as np
import matplotlib.pyplot as plt

# === Цільова функція (варіант 1) ===
def target_function(x):
    x1, x2, x3 = x
    return x1**2 - x2**2 + x3**2

base_x = np.array([1, 2, 3])

# === Генерація датасету з шумом і кроком ±1 ===
def generate_dataset(n_samples, noise_std=0.0):
    X, Y = [], []
    while len(X) < n_samples:
        delta = np.random.choice([-1, 0, 1], size=3)
        x = base_x + delta + np.random.normal(0, noise_std, 3)
        X.append(x)
        Y.append(target_function(x))
    return np.array(X), np.array(Y)

# навчальна та контрольна вибірки
X_train, Y_train = generate_dataset(30)
X_test, Y_test = generate_dataset(20)

# нормування
X_min, X_max = X_train.min(axis=0), X_train.max(axis=0)
Y_min, Y_max = Y_train.min(), Y_train.max()

Xn_train = (X_train - X_min) / (X_max - X_min)
Xn_test = (X_test - X_min) / (X_max - X_min)

Yn_train = (Y_train - Y_min) / (Y_max - Y_min)
Yn_test = (Y_test - Y_min) / (Y_max - Y_min)

# === RBF функція та мережа ===
def rbf(x, c, sigma):
    return np.exp(-np.linalg.norm(x - c)**2 / (2 * sigma**2))

class RBFNetwork:
    def __init__(self, n_centers, sigma):
        self.n_centers = n_centers
        self.sigma = sigma

    def fit(self, X, Y):
        idx = np.random.choice(len(X), self.n_centers, replace=False)
        self.centers = X[idx]
        Phi = np.array([[rbf(x, c, self.sigma) for c in self.centers] for x in X])
        self.W = np.linalg.pinv(Phi) @ Y

    def predict(self, X):
        Phi = np.array([[rbf(x, c, self.sigma) for c in self.centers] for x in X])
        return Phi @ self.W

# === Підбір ширини активаційного вікна ===
sigmas = np.linspace(0.1, 1.5, 10)
errors = []

for s in sigmas:
    rbf_net = RBFNetwork(n_centers=10, sigma=s)
    rbf_net.fit(Xn_train, Yn_train)
    Y_pred = rbf_net.predict(Xn_test)
    errors.append(np.mean((Yn_test - Y_pred)**2))

plt.figure()
plt.plot(sigmas, errors, marker='o')
plt.xlabel("Ширина активаційного вікна σ")
plt.ylabel("MSE")
plt.title("Вплив ширини вікон RBF")
plt.grid(True)
plt.show()

# === Вплив кількості навчальних шаблонів ===
train_sizes = [10, 20, 30, 40]
errors_train = []

for n in train_sizes:
    Xtr, Ytr = generate_dataset(n)
    Xtr_n = (Xtr - X_min) / (X_max - X_min)
    Ytr_n = (Ytr - Y_min) / (Y_max - Y_min)
    rbf_net = RBFNetwork(n_centers=min(10, n), sigma=0.6)
    rbf_net.fit(Xtr_n, Ytr_n)
    Y_pred = rbf_net.predict(Xn_test)
    errors_train.append(np.mean((Yn_test - Y_pred)**2))

plt.figure()
plt.plot(train_sizes, errors_train, marker='o')
plt.xlabel("Кількість навчальних шаблонів")
plt.ylabel("MSE")
plt.title("Якість апроксимації vs обсяг навчальної вибірки")
plt.grid(True)
plt.show()

# === Вплив дисперсії навчальних даних ===
noise_levels = [0.0, 0.1, 0.2, 0.4]
errors_noise = []

for noise in noise_levels:
    Xtr, Ytr = generate_dataset(30, noise_std=noise)
    Xtr_n = (Xtr - X_min) / (X_max - X_min)
    Ytr_n = (Ytr - Y_min) / (Y_max - Y_min)
    rbf_net = RBFNetwork(n_centers=10, sigma=0.6)
    rbf_net.fit(Xtr_n, Ytr_n)
    Y_pred = rbf_net.predict(Xn_test)
    errors_noise.append(np.mean((Yn_test - Y_pred)**2))

plt.figure()
plt.plot(noise_levels, errors_noise, marker='o')
plt.xlabel("Дисперсія (рівень шуму)")
plt.ylabel("MSE")
plt.title("Вплив дисперсії на якість RBF")
plt.grid(True)
plt.show()

# === Інтерполяція vs екстраполяція ===
# Інтерполяція (всередині області)
X_interp, Y_interp = generate_dataset(20)
X_interp_n = (X_interp - X_min) / (X_max - X_min)

# Екстраполяція (поза областю)
X_extra = base_x + np.random.uniform(2, 4, (20,3))
X_extra_n = (X_extra - X_min) / (X_max - X_min)
Y_extra = np.array([target_function(x) for x in X_extra])
Y_extra_n = (Y_extra - Y_min) / (Y_max - Y_min)

rbf_net = RBFNetwork(n_centers=10, sigma=0.6)
rbf_net.fit(Xn_train, Yn_train)

err_interp = np.mean((Y_interp - rbf_net.predict(X_interp_n))**2)
err_extra = np.mean((Y_extra_n - rbf_net.predict(X_extra_n))**2)

print("MSE інтерполяції:", err_interp)
print("MSE екстраполяції:", err_extra)
