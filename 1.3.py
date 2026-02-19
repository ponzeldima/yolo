import numpy as np
from scipy.stats import chi2, f, t

# ======================
# ВИХІДНІ ДАНІ (табл. 1.8)
# ======================
X1 = np.array([1, 5, 12, 23, 34, 53, 66, 69, 78], dtype=float)
X2 = np.array([88, 77, 66, 56, 43, 34, 31, 23, 22], dtype=float)
X3 = np.array([11, 32, 34, 45, 48, 65, 77, 88, 96], dtype=float)

X = np.column_stack((X1, X2, X3))
n, p = X.shape   # n = 9, p = 3

print("\n1.3 ТЕСТ ФАРРАРА–ГЛОБЕРА")

# ======================
# НОРМУВАННЯ ФАКТОРІВ
# ======================
X_mean = X.mean(axis=0)
X_std  = X.std(axis=0, ddof=1)
Z = (X - X_mean) / X_std

# ======================
# КОРЕЛЯЦІЙНА МАТРИЦЯ
# ======================
R = (Z.T @ Z) / n
print("\nКореляційна матриця R:\n", R)

# ======================
# ГЛОБАЛЬНИЙ χ²-ТЕСТ
# ======================
det_R = np.linalg.det(R)
chi2_stat = -(n - 1 - (2*p + 5)/6) * np.log(det_R)
df_chi2 = p * (p - 1) // 2
chi2_crit = chi2.ppf(0.95, df_chi2)

print(f"\nχ² = {chi2_stat:.4f}")
print(f"χ²_кр = {chi2_crit:.4f}")

if chi2_stat > chi2_crit:
    print("Висновок: глобальна мультиколінеарність ПРИСУТНЯ.")
else:
    print("Висновок: мультиколінеарність НЕ виявлена.")

# ======================
# F-ТЕСТ ДЛЯ ОКРЕМИХ ФАКТОРІВ
# ======================
R_inv = np.linalg.inv(R)
df1 = p - 1
df2 = n - p
F_crit = f.ppf(0.95, df1, df2)

print(f"\nF_кр = {F_crit:.4f}")
print("\nF-статистики для факторів:")

for i in range(p):
    R_i = np.delete(R, i, axis=0)
    R_i = np.delete(R_i, i, axis=1)
    F_i = ((np.linalg.det(R_i) / det_R) ** (1 / (p - 1)) - 1) * (df2 / df1)

    print(f"F[{i+1}] = {F_i:.4f}", end=" → ")

    if F_i > F_crit:
        print("мультиколінеарний фактор")
    else:
        print("фактор прийнятний")

# ======================
# ЧАСТКОВІ КОРЕЛЯЦІЇ ТА t-ТЕСТ
# ======================
partial_corr = np.zeros((p, p))
t_stats = np.zeros((p, p))
t_crit = t.ppf(1 - 0.05 / 2, df2)

print(f"\nt_кр = {t_crit:.4f}")
print("\nt-статистики часткових кореляцій:")

for i in range(p):
    for j in range(p):
        if i != j:
            r_ij = -R_inv[i, j] / np.sqrt(R_inv[i, i] * R_inv[j, j])
            t_val = r_ij * np.sqrt(df2 / (1 - r_ij**2))
            t_stats[i, j] = t_val

            print(f"t[{i+1},{j+1}] = {t_val:.4f}", end=" → ")

            if abs(t_val) > t_crit:
                print("значуща кореляція (мультиколінеарність)")
            else:
                print("незначуща кореляція")
