# ===============================
# Завдання 6.4
# Кластеризація за допомогою пірамідальної мережі, що росте
# ===============================

# ---------- Дані ----------
X = [
    [1, 2, 'A', 2, 3, 'C'],  # O1
    [1, 4, 'B', 3, 2, 'C'],  # O2
    [3, 4, 'A', 2, 4, 'D'],  # O3
    [2, 2, 'B', 2, 3, 'D'],  # O4
    [1, 2, 'B', 2, 3, 'D'],  # O5
    [3, 4, 'B', 3, 2, 'C'],  # O6
    [1, 4, 'B', 3, 2, 'C'],  # O7
    [1, 4, 'A', 2, 3, 'D']   # O8
]

objects = ['O1','O2','O3','O4','O5','O6','O7','O8']

# ---------- Формування вузлів за ознакою X6 ----------
V1, V2 = [], []

for i, obj in enumerate(X):
    if obj[5] == 'C':
        V1.append(i)
    else:
        V2.append(i)

def summarize_node(indices, X):
    node_summary = []
    for j in range(6):
        node_summary.append(set(X[i][j] for i in indices))
    return node_summary

node_V1 = summarize_node(V1, X)
node_V2 = summarize_node(V2, X)

print("Вузол V1 (X6 = C):", node_V1)
print("Вузол V2 (X6 = D):", node_V2)

# ---------- Верхній концептор ----------
top_conceptor = {'X6': {'C', 'D'}}
print("Верхній концептор:", top_conceptor)

# ---------- Тестові приклади ----------
test_objects = [
    [1, 3, 'A', 2, 3, 'C'],  # O_test_1
    [2, 3, 'A', 3, 4, 'C']   # O_test_2
]

def classify(obj, nodes, node_names):
    for idx, node in enumerate(nodes):
        match = True
        for j in range(6):
            if obj[j] not in node[j]:
                match = False
                break
        if match:
            return node_names[idx]
    return "Нове поняття"

nodes = [node_V1, node_V2]
node_names = ['V1', 'V2']

for i, test_obj in enumerate(test_objects, 1):
    result = classify(test_obj, nodes, node_names)
    print(f"O_test_{i} віднесено до: {result}")

# ---------- Висновок ----------
print("\nВисновок:")
print("Пірамідальна мережа дозволяє формувати узагальнені поняття,")
print("класифікувати нові об'єкти та ініціювати уточнення моделей при")
print("появі об'єктів, що не узгоджуються з існуючими поняттями.")
