import numpy as np
import matplotlib.pyplot as plt

# === Цільова функція (варіант 1) ===
def target_function(x):
    x1, x2, x3 = x
    return x1**2 - x2**2 + x3**2

# базова точка (варіант 1)
base_x = np.array([1, 2, 3])

# === Генерація датасету з 20 варіацій (крок ±1) ===
def generate_dataset(n_samples):
    X, Y = [], []
    while len(X) < n_samples:
        delta = np.random.choice([-1, 0, 1], size=3)
        x = base_x + delta
        X.append(x)
        Y.append(target_function(x))
    return np.array(X), np.array(Y)

X_all, Y_all = generate_dataset(20)

# нормування
X_min, X_max = X_all.min(axis=0), X_all.max(axis=0)
Y_min, Y_max = Y_all.min(), Y_all.max()

Xn = (X_all - X_min) / (X_max - X_min)
Yn = (Y_all - Y_min) / (Y_max - Y_min)

Y_mean = Yn.mean()

# виходи: d1 — значення, d2 — класифікація за середнім
D = np.column_stack([
    Yn,
    (Yn > Y_mean).astype(int)
])

# train / test
X_train, X_test = Xn[:14], Xn[14:]
D_train, D_test = D[:14], D[14:]

# === Sigmoid функція ===
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(y):
    return y * (1 - y)

# === Нейромережа з навчанням методом зворотного поширення ===
class BackpropNN:
    def __init__(self, n_input, n_hidden, n_output=2, eta=0.1):
        self.W1 = np.random.uniform(-1, 1, (n_input, n_hidden))
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = np.random.uniform(-1, 1, (n_hidden, n_output))
        self.b2 = np.zeros((1, n_output))
        self.eta = eta

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.y = sigmoid(self.z2)
        return self.y

    def train(self, X, D, epochs=2000):
        for _ in range(epochs):
            Y = self.forward(X)
            error = D - Y
            d2 = error * sigmoid_derivative(Y)
            d1 = d2 @ self.W2.T * sigmoid_derivative(self.a1)

            self.W2 += self.eta * self.a1.T @ d2
            self.b2 += self.eta * d2.sum(axis=0)
            self.W1 += self.eta * X.T @ d1
            self.b1 += self.eta * d1.sum(axis=0)

    def accuracy(self, X, D, threshold=0.5):
        Y = self.forward(X)
        pred = (Y[:,1] > threshold).astype(int)
        true = D[:,1].astype(int)
        return np.mean(pred == true)

# === 1. Ймовірність правильної відповіді vs кількість нейронів прихованого шару ===
hidden_sizes = [2, 3, 5, 7, 10]
acc_hidden = []

for h in hidden_sizes:
    nn = BackpropNN(n_input=3, n_hidden=h)
    nn.train(X_train, D_train)
    acc_hidden.append(nn.accuracy(X_test, D_test))

plt.figure()
plt.plot(hidden_sizes, acc_hidden, marker='o')
plt.xlabel("Кількість нейронів у прихованому шарі")
plt.ylabel("Ймовірність правильної відповіді")
plt.title("Точність vs прихований шар")
plt.grid(True)
plt.show()

# === 2. Ймовірність правильної відповіді vs кількість вхідних нейронів ===
input_sizes = [1, 2, 3]
acc_input = []

for k in input_sizes:
    nn = BackpropNN(n_input=k, n_hidden=5)
    nn.train(X_train[:, :k], D_train)
    acc_input.append(nn.accuracy(X_test[:, :k], D_test))

plt.figure()
plt.plot(input_sizes, acc_input, marker='o')
plt.xlabel("Кількість вхідних нейронів")
plt.ylabel("Ймовірність правильної відповіді")
plt.title("Точність vs вхідний шар")
plt.grid(True)
plt.show()

# === 3. Ймовірність правильної відповіді vs розмір навчальної вибірки ===
train_sizes = [5, 10, 14]
acc_train = []

for n in train_sizes:
    nn = BackpropNN(n_input=3, n_hidden=5)
    nn.train(X_train[:n], D_train[:n])
    acc_train.append(nn.accuracy(X_test, D_test))

plt.figure()
plt.plot(train_sizes, acc_train, marker='o')
plt.xlabel("Кількість навчальних прикладів")
plt.ylabel("Ймовірність правильної відповіді")
plt.title("Точність vs обсяг навчальної вибірки")
plt.grid(True)
plt.show()

# === 4. Ймовірність правильної відповіді vs поріг нейрона 1 ===
thresholds = np.linspace(0.2, 0.8, 7)
acc_threshold = []

nn = BackpropNN(n_input=3, n_hidden=5)
nn.train(X_train, D_train)

for t in thresholds:
    acc_threshold.append(nn.accuracy(X_test, D_test, threshold=t))

plt.figure()
plt.plot(thresholds, acc_threshold, marker='o')
plt.xlabel("Порогове значення")
plt.ylabel("Ймовірність правильної відповіді")
plt.title("Точність vs поріг нейрона")
plt.grid(True)
plt.show()
