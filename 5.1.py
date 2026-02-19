import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster

# =====================================
# 1. Вхідні дані (таблиця 5.4)
# =====================================
data = np.array([
    [71.2, 54.7, 128, 38995, 10.43, 412.8, 3436, 101.6, 21.0, 17.9, 287],
    [71.6, 55.56, 120.1, 13636, 14.92, 452.7, 3899, 130.408, 19.0, 18.26, 282.56],
    [74.25, 55.49, 114.4, 12905, 12.11, 410.2, 4644, 101.306, 18.0, 18.71, 281.05],
    [74.25, 56.12, 113.9, 13271, 18.67, 458.6, 2051, 141.993, 26.0, 19.69, 283.68],
    [78.38, 61.78, 116.8, 26785, 56.83, 518.6, 2562, 385.409, 36.0, 20.19, 299.65],
    [82.2, 64.22, 115.3, 30437, 80.62, 555.9, 2855, 482.03, 15.3, 20.42, 307.13],
    [84.28, 65.32, 120.1, 42156, 18.65, 458.6, 2855, 141.932, 16.25, 20.89, 316.85],
    [86.08, 68.48, 121.0, 12936, 34.30, 483.1, 2891, 222.11, 17.64, 20.90, 324.33],
    [87.94, 71.8, 125.8, 23894, 41.60, 494.6, 3512, 274.297, 19.23, 21.48, 336.90],
    [89.43, 74.24, 127.8, 25355, 34.31, 483.2, 1251, 222.196, 19.74, 21.53, 343.80],
    [104.6, 98.9, 147.9, 38759, 28.2, 473.6, 3299, 185.061, 26.45, 26.83, 421.71]
])

# =====================================
# 2. Формування X та Y
# =====================================
X = data[:, :-1]
Y = data[:, -1]

features = list(range(X.shape[1]))

# =====================================
# 3. Розбиття на навчальну і тестову вибірки
# =====================================
split_idx = int(0.7 * len(X))

X_train, X_test = X[:split_idx], X[split_idx:]
Y_train, Y_test = Y[:split_idx], Y[split_idx:]

# =====================================
# 4. Функція обчислення MSE
# =====================================
def mse_for_features(idx):
    Xtr = np.column_stack([np.ones(len(X_train)), X_train[:, idx]])
    Xte = np.column_stack([np.ones(len(X_test)), X_test[:, idx]])

    coef = np.linalg.lstsq(Xtr, Y_train, rcond=None)[0]
    Y_pred = Xte @ coef

    return np.mean((Y_test - Y_pred) ** 2)

# =====================================
# 5. Del — метод виключення
# =====================================
def method_del(feats, k):
    cur = feats.copy()
    while len(cur) > k:
        errors = []
        for f in cur:
            subset = [x for x in cur if x != f]
            errors.append((f, mse_for_features(subset)))
        worst = min(errors, key=lambda x: x[1])[0]
        cur.remove(worst)
    return cur

# =====================================
# 6. Add — метод додавання
# =====================================
def method_add(feats, k):
    start = min(feats, key=lambda f: mse_for_features([f]))
    selected = [start]

    while len(selected) < k:
        scores = []
        for f in feats:
            if f not in selected:
                scores.append((f, mse_for_features(selected + [f])))
        selected.append(min(scores, key=lambda x: x[1])[0])
    return selected

# =====================================
# 7. Комбіновані методи
# =====================================
def method_add_del(k):
    temp = method_add(features, k + 2)
    return method_del(temp, k)

def method_del_add(k):
    temp = method_del(features, k + 2)
    return method_add(temp, k)

# =====================================
# 8. ВПА — випадковий пошук
# =====================================
def method_vpa(k, iters=50):
    best_mse = np.inf
    best_set = None

    for _ in range(iters):
        subset = list(np.random.choice(features, k, replace=False))
        err = mse_for_features(subset)
        if err < best_mse:
            best_mse = err
            best_set = subset
    return best_set

# =====================================
# 9. НТПО — кластеризація факторів
# =====================================
def method_ntpo(k):
    corr = np.corrcoef(X.T)
    Z = linkage(1 - np.abs(corr), method="average")
    clusters = fcluster(Z, t=k, criterion="maxclust")

    reps = []
    for c in np.unique(clusters):
        reps.append(np.where(clusters == c)[0][0])

    return method_del(reps, k)

# =====================================
# 10. Запуск і порівняння
# =====================================
methods = {
    "Del": method_del(features, 5),
    "Add": method_add(features, 5),
    "AddDel": method_add_del(5),
    "DelAdd": method_del_add(5),
    "VPA": method_vpa(5),
    "NTPO": method_ntpo(5)
}

print("Метод   | Інформативні фактори | MSE")
print("-" * 50)

for name, idx in methods.items():
    mse = mse_for_features(idx)
    labels = [f"X{i+1}" for i in idx]
    print(f"{name:7s}| {labels} | {mse:.4f}")
