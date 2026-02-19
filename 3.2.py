import numpy as np
import random

# --- Параметри варіанту 1 ---
DIMENSIONS = 2
X_MIN, X_MAX = -1.28, 1.28
BIT_DEPTH = 16  # точність кодування кожного x_i
POP_SIZE = 50
GEN_COUNT = 100
P_MUTATION = 0.01

def fitness_function(x_vars):
    """Цільова функція (мінімізація)"""
    x1, x2 = x_vars
    denom = 100.0 * (x1 * x1 - x2) ** 2 + (1.0 - x1) ** 2 + 1.0
    return 100.0 / denom

def decode(chromosome):
    """Декодування бінарної хромосоми у дійсні числа"""
    decoded_vars = []
    chunk_size = BIT_DEPTH
    for i in range(DIMENSIONS):
        chunk = chromosome[i*chunk_size : (i+1)*chunk_size]
        binary_str = "".join(map(str, chunk))
        val_int = int(binary_str, 2)
        # Масштабування в діапазон [X_MIN, X_MAX]
        val_real = X_MIN + val_int * (X_MAX - X_MIN) / (2**BIT_DEPTH - 1)
        decoded_vars.append(val_real)
    return np.array(decoded_vars)

def crossover_double_point(parent1, parent2):
    """Двоточковий кросовер (тип D)"""
    size = len(parent1)
    pt1, pt2 = sorted(random.sample(range(size), 2))
    child1 = np.concatenate([parent1[:pt1], parent2[pt1:pt2], parent1[pt2:]])
    child2 = np.concatenate([parent2[:pt1], parent1[pt1:pt2], parent2[pt2:]])
    return child1, child2

def crossover_one_point(parent1, parent2):
    """Одноточковий кросовер (тип O)"""
    size = len(parent1)
    pt = random.randint(1, size - 1) # Вибираємо одну точку
    child1 = np.concatenate([parent1[:pt], parent2[pt:]])
    child2 = np.concatenate([parent2[:pt], parent1[pt:]])
    return child1, child2

def mutate(chromosome, prob):
    """Мутація з ймовірністю Pm"""
    for i in range(len(chromosome)):
        if random.random() < prob:
            chromosome[i] = 1 - chromosome[i]
    return chromosome

# --- Основний цикл ГА ---

# 1. Початкова популяція
chrom_len = DIMENSIONS * BIT_DEPTH
population = [np.random.randint(0, 2, chrom_len) for _ in range(POP_SIZE)]

for gen in range(GEN_COUNT):
    # 1. Обчислюємо пристосованість кожної особини
    scores = []
    for chrom in population:
        x_vals = decode(chrom)
        scores.append(fitness_function(x_vals))
    
    # 2. ЕЛІТАРНІСТЬ: Знаходимо та копіюємо найкращу особину поточного покоління
    best_idx = np.argmax(scores)
    elite_individual = population[best_idx].copy()
    current_best_score = scores[best_idx]
    
    new_population = []
    
    # 3. Формуємо нове покоління
    while len(new_population) < POP_SIZE:
        # ПАНМІКСІЯ: Вибираємо двох батьків абсолютно випадково з усієї популяції
        p1 = random.choice(population)
        p2 = random.choice(population)
        
        # ОДНОТОЧКОВИЙ КРОСОВЕР
        c1, c2 = crossover_one_point(p1, p2)
        
        # МУТАЦІЯ та додавання до нової популяції
        new_population.append(mutate(c1, P_MUTATION))
        if len(new_population) < POP_SIZE:
            new_population.append(mutate(c2, P_MUTATION))
            
    # 4. ЗАСТОСУВАННЯ ЕЛІТАРНОСТІ: 
    # Замінюємо першу особину нової популяції на нашу збережену "еліту"
    new_population[0] = elite_individual
    
    # Оновлюємо популяцію
    population = new_population

    # Вивід прогресу кожні 10 поколінь
    if gen % 10 == 0:
        print(f"Покоління {gen}: Найкраща пристосованість = {current_best_score:.6f}")
        
        
# Результат
best_idx = np.argmax(scores)
best_x = decode(population[best_idx])
print(f"Найкращий результат після {GEN_COUNT} поколінь:")
print(f"X: {best_x}")
print(f"F(X): {fitness_function(best_x)}")