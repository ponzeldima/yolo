import numpy as np

# ===============================
# 6.2 ПОРІВНЯЛЬНИЙ АНАЛІЗ
# Класичні vs еволюційна кластеризація
# ===============================

# ---------- Дані ----------
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
Xn = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

K = 2

# ---------- Еволюційна кластеризація ----------
def genetic_clustering(X, k, pop=30, gens=200, pm=0.3, sigma=0.1, seed=1):
    np.random.seed(seed)
    n, m = X.shape
    chrom_len = k * m

    def J_value(ch):
        centers = ch.reshape(k, m)
        d = ((X[:, None] - centers)**2).sum(axis=2)
        lbl = np.argmin(d, axis=1)
        return ((X - centers[lbl])**2).sum()

    population = np.random.rand(pop, chrom_len)

    for _ in range(gens):
        J_vals = np.array([J_value(ind) for ind in population])
        best_half = population[np.argsort(J_vals)[:pop // 2]]

        children = []
        while len(children) < pop:
            p1, p2 = best_half[np.random.choice(len(best_half), 2, replace=False)]
            cut = np.random.randint(1, chrom_len - 1)
            child = np.concatenate((p1[:cut], p2[cut:]))
            children.append(child)

        children = np.array(children)

        mut_mask = np.random.rand(*children.shape) < pm
        children[mut_mask] += np.random.normal(0, sigma, mut_mask.sum())
        children = np.clip(children, 0, 1)

        population = children

    J_vals = np.array([J_value(ind) for ind in population])
    best = population[np.argmin(J_vals)]
    best_J = J_vals.min()

    centers = best.reshape(k, m)
    dist = ((X[:, None] - centers)**2).sum(axis=2)
    labels = np.argmin(dist, axis=1)
    sizes = [np.sum(labels == i) for i in range(k)]

    return best_J, sizes

# ---------- Один запуск ----------
J_best, sizes = genetic_clustering(Xn, K)
print("\nЕВОЛЮЦІЙНА КЛАСТЕРИЗАЦІЯ")
print("Наповнення кластерів:", sizes)
print("Цільова функція J =", round(J_best, 4))

# ---------- Перевірка стійкості ----------
def add_noise(X, eps, seed):
    np.random.seed(seed)
    return np.clip(X + np.random.normal(0, eps, X.shape), 0, 1)

eps = 0.02
runs = 5
J_vals = []

for i in range(runs):
    Xr = add_noise(Xn, eps, seed=100+i)
    J_run, _ = genetic_clustering(Xr, K, seed=200+i)
    J_vals.append(J_run)

J_vals = np.array(J_vals)

print("\nСТІЙКІСТЬ ДО ЗБУРЕНЬ")
print("J по запусках:", [round(v,4) for v in J_vals])
print("Середнє J =", round(J_vals.mean(), 4))
print("Ст. відхилення =", round(J_vals.std(ddof=1), 4))

# ---------- Вплив параметра POP_SIZE ----------
print("\nВПЛИВ POP_SIZE НА ТОЧНІСТЬ")
for p in [20, 40, 60]:
    Jp, _ = genetic_clustering(Xn, K, pop=p)
    print(f"POP_SIZE = {p:>2d} -> J = {Jp:.4f}")
