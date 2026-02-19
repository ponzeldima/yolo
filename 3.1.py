import numpy as np
import pandas as pd
from itertools import product

# --- 1. НАЛАШТУВАННЯ ВАРІАНТА 2 ---
BASE_X = np.array([7, 3, 8]) 
Y_TARGET = 13

def true_function(x):
    # Тестова логіка (можна змінювати коефіцієнти)
    return 10 + 0.5*x[0] - 2.1*x[1] + 0.08*(x[2]**2)

bias = Y_TARGET - true_function(BASE_X)
def get_y(x): return true_function(x) + bias

# Генерація даних
rows = [[*p, get_y(p)] for p in [BASE_X + np.array(d) for d in product([-1,0,1], repeat=3)]]
df = pd.DataFrame(rows, columns=['x1', 'x2', 'x3', 'y'])

# Поділ на Навчальну (A) та Контрольну (B) вибірки
train_df = df.sample(frac=0.6, random_state=42)
test_df = df.drop(train_df.index)

# --- 2. МНК ТА БАГАТОРИДНИЙ АЛГОРИТМ ---
def gmdh_multilayer_reg(df_tr, df_ts, F=3):
    target = 'y'
    # Початкові назви ознак та їх значення
    current_features = ['x1', 'x2', 'x3']
    current_tr_x = df_tr[current_features].values
    current_ts_x = df_ts[current_features].values
    
    # Словник для зберігання текстових формул кожної ознаки
    feature_formulas = {i: name for i, name in enumerate(current_features)}
    
    best_err = float('inf')
    best_model = None
    layer = 1

    while True:
        layer_results = []
        n = current_tr_x.shape[1]
        
        for i in range(n):
            for j in range(i + 1, n):
                # МНК для: y = a0 + a1*xi + a2*xj
                Xi_tr, Xj_tr = current_tr_x[:, i], current_tr_x[:, j]
                A = np.column_stack([np.ones(len(Xi_tr)), Xi_tr, Xj_tr])
                w = np.linalg.pinv(A.T @ A) @ A.T @ df_tr[target].values
                
                # Перевірка Регулярності (на вибірці B)
                Xi_ts, Xj_ts = current_ts_x[:, i], current_ts_x[:, j]
                preds_ts = w[0] + w[1]*Xi_ts + w[2]*Xj_ts
                error = np.mean((df_ts[target].values - preds_ts)**2)
                
                # Формуємо текст нової формули
                f_i = feature_formulas[i]
                f_j = feature_formulas[j]
                text_form = f"({w[0]:.3f} + {w[1]:.3f}*{f_i} + {w[2]:.3f}*{f_j})"
                
                layer_results.append({
                    'error': error, 'w': w, 'formula': text_form,
                    'out_tr': w[0] + w[1]*Xi_tr + w[2]*Xj_tr,
                    'out_ts': preds_ts
                })
        
        layer_results.sort(key=lambda x: x['error'])
        curr_best_err = layer_results[0]['error']
        
        print(f"Шар {layer}: MSE = {curr_best_err:.6f}")
        
        # Перевірка зупинки (критерій селекції)
        if curr_best_err >= best_err:
            print(">>> Селекція зупинена (мінімум знайдено)")
            break
            
        best_err = curr_best_err
        best_model = layer_results[0]
        
        # Оновлення даних для наступного шару (беремо F найкращих)
        top = layer_results[:F]
        current_tr_x = np.column_stack([m['out_tr'] for m in top])
        current_ts_x = np.column_stack([m['out_ts'] for m in top])
        # Оновлюємо текстові описи ознак для "матрьошки"
        feature_formulas = {idx: m['formula'] for idx, m in enumerate(top)}
        
        layer += 1
        if layer > 5: break

    return best_model

# --- 3. ЗАПУСК ТА ВИВІД ---
result = gmdh_multilayer_reg(train_df, test_df)

print("\n" + "="*30)
print(f"Обрана модель: {result['formula']}")
print(f"Критерій регулярності: {result['error']:.6f}")
print("="*30)