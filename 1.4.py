import numpy as np
import statsmodels.api as sm
from scipy.stats import f

# ======================
# ВИХІДНІ ДАНІ (табл. 1.8)
# ======================
X1 = np.array([1, 5, 12, 23, 34, 53, 66, 69, 78], dtype=float)
X2 = np.array([88, 77, 66, 56, 43, 34, 31, 23, 22], dtype=float)
X3 = np.array([11, 32, 34, 45, 48, 65, 77, 88, 96], dtype=float)
Y  = np.array([2, 4, 8, 12, 17, 32, 54, 65, 77], dtype=float)

X = np.column_stack((X1, X2, X3))
n, p = X.shape

print("\n1.4 ПАРАМЕТРИЧНИЙ ТЕСТ ГОЛЬДФЕЛЬДА–КВАНДТА")

# ======================
# УПОРЯДКУВАННЯ СПОСТЕРЕЖЕНЬ (за X1)
# ======================
order = np.argsort(X1)
X_sorted = X[order]
Y_sorted = Y[order]

# ======================
# ВИКЛЮЧЕННЯ СЕРЕДНІХ СПОСТЕРЕЖЕНЬ
# ======================
k = 2  # кількість виключених центральних спостережень
low_idx  = slice(0, (n - k) // 2)
high_idx = slice((n + k) // 2, n)

X_low,  Y_low  = X_sorted[low_idx],  Y_sorted[low_idx]
X_high, Y_high = X_sorted[high_idx], Y_sorted[high_idx]

# ======================
# ФУНКЦІЯ ОБЧИСЛЕННЯ RSS
# ======================
def calc_rss(Xg, Yg):
    Xg_const = sm.add_constant(Xg)
    beta = np.linalg.inv(Xg_const.T @ Xg_const) @ (Xg_const.T @ Yg)
    resid = Yg - Xg_const @ beta
    return np.sum(resid**2)

RSS1 = calc_rss(X_low,  Y_low)
RSS2 = calc_rss(X_high, Y_high)

# ======================
# F-СТАТИСТИКА ГОЛЬДФЕЛЬДА–КВАНДТА
# ======================
n1, n2 = len(Y_low), len(Y_high)
m = p + 1   # кількість параметрів (фактори + константа)

F_stat = (RSS2 / (n2 - m)) / (RSS1 / (n1 - m))

df1 = n2 - m
df2 = n1 - m
F_crit = f.ppf(0.95, df1, df2)

print(f"\nRSS₁ = {RSS1:.4f}")
print(f"RSS₂ = {RSS2:.4f}")
print(f"F = {F_stat:.4f}")
print(f"F_кр = {F_crit:.4f}")

# ======================
# ВИСНОВОК
# ======================
if F_stat > F_crit:
    print
