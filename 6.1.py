import numpy as np

# ===============================
# 6.1 КЛАСИЧНА КЛАСТЕРИЗАЦІЯ
# ===============================

# ---------- Вхідні дані ----------
X = np.array([
    [6,150,1.8,24,30,120,3.4,15],
    [7,150,1.8,24,30,120,9.7,5],
    [6,170,1.8,24,30,120,7.4,23],
    [7,170,1.8,24,30,120,10.6,8],
    [6,150,2.4,24,30,120,6.5,20],
    [7,150,2.4,24,30,120,7.9,9],
    [6,170,2.4,24,30,120,10.3,13],
    [7,170,2.4,24,30,120,9.5,5],
    [6,150,1.8,36,30,120,14.3,23],
    [7,150,1.8,36,30,120,10.5,1],
    [6,170,1.8,36,30,120,7.8,11],
    [7,170,1.8,36,30,120,17.2,5],
    [6,150,2.4,36,30,120,9.4,15],
    [7,150,2.4,36,30,120,12.1,8],
    [6,170,2.4,36,30,120,9.5,15],
    [7,170,2.4,36,30,120,15.8,1],
    [6,150,1.8,24,42,120,8.3,22],
    [7,150,1.8,24,42,120,8.0,8],
    [6,170,1.8,24,42,120,7.9,16],
    [7,170,1.8,24,42,120,10.7,7],
    [6,150,2.4,24,42,120,7.2,25],
    [7,150,2.4,24,42,120,7.2,5],
    [6,170,2.4,24,42,120,7.9,17],
    [7,170,2.4,24,42,120,10.2,8],
    [6,150,1.8,36,42,120,10.3,10],
    [7,150,1.8,36,42,120,9.9,3],
    [6,170,1.8,36,42,120,7.4,22],
    [7,170,1.8,36,42,120,10.5,6],
    [6,150,2.4,36,42,120,9.6,24],
    [7,150,2.4,36,42,120,15.1,4],
    [6,170,2.4,36,42,120,8.7,10],
    [7,170,2.4,36,42,120,12.1,5],
    [6,150,1.8,24,30,130,12.6,32],
    [7,150,1.8,24,30,130,10.5,10],
    [6,170,1.8,24,30,130,11.3,28],
    [7,170,1.8,24,30,130,10.6,18],
    [6,150,2.4,24,30,130,8.1,22],
    [7,150,2.4,24,30,130,12.5,31],
    [6,170,2.4,24,30,130,11.1,17],
    [7,170,2.4,24,30,130,12.9,16],
    [6,150,1.8,36,30,130,14.6,38],
    [7,150,1.8,36,30,130,12.7,12],
    [6,170,1.8,36,30,130,10.8,34],
    [7,170,1.8,36,30,130,17.1,19],
    [6,150,2.4,36,30,130,13.6,12],
    [7,150,2.4,36,30,130,14.6,14],
    [6,170,2.4,36,30,130,13.3,25],
    [7,170,2.4,36,30,130,14.4,16],
    [6,150,1.8,24,42,130,11.0,31],
    [7,150,1.8,24,42,130,12.5,14],
    [6,170,1.8,24,42,130,8.9,23],
    [7,170,1.8,24,42,130,13.1,23],
    [6,150,2.4,24,42,130,7.6,28],
    [7,150,2.4,24,42,130,8.6,20],
    [6,170,2.4,24,42,130,11.8,18],
    [7,170,2.4,24,42,130,12.4,11]
])

# ---------- Нормування ----------
Xn = (X - X.mean(axis=0)) / X.std(axis=0)

# ---------- K-means ----------
def kmeans_custom(data, k, max_iter=150):
    rng = np.random.default_rng(2)
    centers = data[rng.choice(len(data), k, replace=False)]

    for _ in range(max_iter):
        dist = np.linalg.norm(data[:, None] - centers[None, :], axis=2)
        labels = np.argmin(dist, axis=1)
        new_centers = np.array([data[labels == i].mean(axis=0) for i in range(k)])

        if np.allclose(centers, new_centers):
            break
        centers = new_centers

    return labels, centers

# ---------- Ward ----------
def ward_clustering(data, k):
    clusters = [[i] for i in range(len(data))]

    def ward_metric(a, b):
        A, B = data[a], data[b]
        na, nb = len(A), len(B)
        return (na * nb) / (na + nb) * np.sum((A.mean(0) - B.mean(0))**2)

    while len(clusters) > k:
        best_val = float("inf")
        best_pair = None

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                val = ward_metric(clusters[i], clusters[j])
                if val < best_val:
                    best_val = val
                    best_pair = (i, j)

        i, j = best_pair
        clusters[i].extend(clusters[j])
        clusters.pop(j)

    labels = np.zeros(len(data), dtype=int)
    for idx, cl in enumerate(clusters):
        for el in cl:
            labels[el] = idx

    centers = np.array([data[labels == i].mean(axis=0) for i in range(k)])
    return labels, centers

# ---------- Цільова функція ----------
def objective(data, labels, centers):
    return np.sum((data - centers[labels])**2)

# ---------- Інформативність ----------
def informativeness(data, labels):
    scores = []
    for j in range(data.shape[1]):
        means = [data[labels == i, j].mean() for i in np.unique(labels)]
        scores.append(np.var(means))
    return scores

# ---------- Запуск ----------
K = 2

labels_km, centers_km = kmeans_custom(Xn, K)
labels_w, centers_w = ward_clustering(Xn, K)

print("\nK-MEANS")
print("Наповнення:", [np.sum(labels_km == i) for i in range(K)])
print("J =", round(objective(Xn, labels_km, centers_km), 4))
for i, v in enumerate(informativeness(Xn, labels_km)):
    print(f"X{i+1}: {v:.4f}")

print("\nWARD")
print("Наповнення:", [np.sum(labels_w == i) for i in range(K)])
print("J =", round(objective(Xn, labels_w, centers_w), 4))
for i, v in enumerate(informativeness(Xn, labels_w)):
    print(f"X{i+1}: {v:.4f}")
