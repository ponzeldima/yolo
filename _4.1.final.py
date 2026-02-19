import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from functools import reduce
from itertools import product
from typing import Iterable


@dataclass(frozen=True)
class FuzzyTrapezoid:
    """
    Трапецієподібний нечіткий інтервал, заданий п’ятіркою (_m, m_, a, b, h):
    - _m: ліве модальне значення (початок "ядра"/плато)
    - m_: праве модальне значення (кінець "ядра"/плато)
    - a: лівий коефіцієнт скошеності (довжина лівого плеча)
    - b: правий коефіцієнт скошеності (довжина правого плеча)
    - h: висота (максимальне значення μ), 0 < h <= 1
    """

    m_left: float   # _m
    m_right: float  # m_
    a: float
    b: float
    h: float = 1.0
    name: str = ""

    @property
    def support_left(self) -> float:
        return self.m_left - self.a

    @property
    def support_right(self) -> float:
        return self.m_right + self.b

    @property
    def core_left(self) -> float:
        return self.m_left

    @property
    def core_right(self) -> float:
        return self.m_right


def mu_trapezoid_from_5tuple(x: np.ndarray, t: FuzzyTrapezoid) -> np.ndarray:
    """
    μ(x) для трапецієподібного нечіткого інтервалу (_m, m_, a, b, h).
    """
    y = np.zeros_like(x, dtype=float)

    left0 = t.support_left
    left1 = t.m_left
    right1 = t.m_right
    right0 = t.support_right

    # Лівий нахил: 0 -> h
    if t.a > 0:
        m = (x >= left0) & (x < left1)
        y[m] = t.h * (x[m] - left0) / t.a

    # Плато (ядро): h
    m = (x >= left1) & (x <= right1)
    y[m] = t.h

    # Правий спад: h -> 0
    if t.b > 0:
        m = (x > right1) & (x <= right0)
        y[m] = t.h * (right0 - x[m]) / t.b

    return np.clip(y, 0.0, t.h)


def add_trapezoids(t1: FuzzyTrapezoid, t2: FuzzyTrapezoid, name: str = "") -> FuzzyTrapezoid:
    """
    Додавання двох трапецієподібних нечітких інтервалів за методичкою:

    Нехай Mi = (_m_i, m_i_, a_i, b_i, h_i), Mj = (_m_j, m_j_, a_j, b_j, h_j),
    тоді Mi + Mj = (_m, m_, a, b, h), де:

    h = min(h_i, h_j)
    a = h * (a_i / h_i + a_j / h_j)
    b = h * (b_i / h_i + b_j / h_j)
    _m = _m_i + _m_j - a_i - a_j + a
    m_ = m_i_ + m_j_ + b_i + b_j - b
    """
    if t1.h <= 0 or t2.h <= 0:
        raise ValueError("Heights h must be > 0.")
    if t1.a < 0 or t1.b < 0 or t2.a < 0 or t2.b < 0:
        raise ValueError("Skew coefficients a,b must be >= 0.")

    h = min(t1.h, t2.h)
    a = h * (t1.a / t1.h + t2.a / t2.h)
    b = h * (t1.b / t1.h + t2.b / t2.h)
    m_left = t1.m_left + t2.m_left - t1.a - t2.a + a
    m_right = t1.m_right + t2.m_right + t1.b + t2.b - b

    return FuzzyTrapezoid(m_left=m_left, m_right=m_right, a=a, b=b, h=h, name=name)


@dataclass(frozen=True)
class ChoiceGroup:
    """
    Група альтернатив (OR): у сценарії обирається рівно ОДНА опція з options.
    Напр., В може бути V1 або V2; Г може бути G1 або G2.
    """

    name: str
    options: tuple[FuzzyTrapezoid, ...]


def sum_trapezoids(items: Iterable[FuzzyTrapezoid], name: str) -> FuzzyTrapezoid:
    """Додає багато інтервалів послідовно через add_trapezoids()."""
    items = list(items)
    if not items:
        raise ValueError("Need at least one trapezoid to sum.")
    total = reduce(lambda acc, t: add_trapezoids(acc, t, name=name), items)
    return FuzzyTrapezoid(**{**total.__dict__, "name": name})


def scenario_totals(
    fixed: list[FuzzyTrapezoid],
    groups: list[ChoiceGroup],
    total_name_prefix: str = "S",
) -> list[tuple[str, list[FuzzyTrapezoid], FuzzyTrapezoid]]:
    """
    Генерує всі сценарії:
    - fixed додаються завжди
    - з кожної групи береться рівно одна опція
    Повертає список: (scenario_name, chosen_list, total_trapezoid).
    """
    if not groups:
        t = sum_trapezoids(fixed, name=f"{total_name_prefix}0")
        return [(f"{total_name_prefix}0", fixed, t)]

    scenarios: list[tuple[str, list[FuzzyTrapezoid], FuzzyTrapezoid]] = []
    for idx, choice_tuple in enumerate(product(*[g.options for g in groups]), start=1):
        chosen = list(fixed) + list(choice_tuple)
        s_name = f"{total_name_prefix}{idx}"
        total = sum_trapezoids(chosen, name=s_name)
        scenarios.append((s_name, chosen, total))
    return scenarios


def mu_union(x: np.ndarray, items: Iterable[FuzzyTrapezoid]) -> np.ndarray:
    """Нечітке 'АБО': μ(x) = max(μ1(x), μ2(x), ...)."""
    mus = [mu_trapezoid_from_5tuple(x, t) for t in items]
    if not mus:
        return np.zeros_like(x, dtype=float)
    return np.maximum.reduce(mus)


def get_intervals(x: np.ndarray, mask: np.ndarray) -> list[tuple[float, float]]:
    """Перетворює маску True/False у список інтервалів (по сітці x)."""
    if not np.any(mask):
        return []
    idx = np.where(mask)[0]
    intervals: list[tuple[float, float]] = []
    start = idx[0]
    for i in range(1, len(idx)):
        if idx[i] != idx[i - 1] + 1:
            intervals.append((float(x[start]), float(x[idx[i - 1]])))
            start = idx[i]
    intervals.append((float(x[start]), float(x[idx[-1]])))
    return intervals


def describe(t: FuzzyTrapezoid) -> str:
    return (
        f"{t.name}: (_m={t.m_left:g}, m_={t.m_right:g}, a={t.a:g}, b={t.b:g}, h={t.h:g}) | "
        f"ядро=[{t.core_left:g}, {t.core_right:g}] | "
        f"носій=[{t.support_left:g}, {t.support_right:g}]"
    )


def plot_single(x: np.ndarray, mu: np.ndarray, title: str, label: str, color: str = "black") -> None:
    """Окремий (single) графік для однієї функції належності."""
    plt.figure(figsize=(10, 4))
    plt.plot(x, mu, color=color, linewidth=2.5, alpha=0.85, label=label)
    plt.fill_between(x, mu, color=color, alpha=0.10)
    plt.title(title)
    plt.xlabel("Сума фінансування (у.о.)")
    plt.ylabel("μ(x)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


if __name__ == "__main__":

    # --- Конфігурація (оригінальні дані з умови задачі 4.1) ---
    # 1) Установа A: сума 300 у.о. (чітке значення)
    A = FuzzyTrapezoid(m_left=300, m_right=300, a=0, b=0, h=1, name="A")

    # 2) Установа Б: носій 250..400, ядро 300..350
    B = FuzzyTrapezoid(m_left=300, m_right=350, a=50, b=50, h=1, name="Б")

    # 3) Установа В: 200..300 із зростанням упевненості при зростанні суми
    # Представляємо як трапецію з ядром у точці 300, лівим плечем 100 (до 200), b=0.
    V = FuzzyTrapezoid(m_left=300, m_right=300, a=100, b=0, h=1, name="В")

    # 4) Установа Г: носій 2000..2400, ядро 2100..2200
    G = FuzzyTrapezoid(m_left=2100, m_right=2200, a=100, b=200, h=1, name="Г")

    # 5) Установа Д: "скоріше за все НЕ профінансує", але якщо профінансує — 300..500 зі спаданням.
    # Моделюємо як OR (альтернатива):
    # - Д1: "дасть 300..500" з низькою впевненістю h=0.2
    # - Д2: "не дасть нічого" з впевненістю h=0.8 (нульовий інтервал)
    D1 = FuzzyTrapezoid(m_left=300, m_right=300, a=0, b=200, h=0.2, name="Д1")
    D2 = FuzzyTrapezoid(m_left=0, m_right=0, a=0, b=0, h=0.8, name="Д2")
    D_group = ChoiceGroup(name="Д", options=(D1, D2))

    # Фіксовані джерела: A, Б, В, Г (завжди беруть участь)
    fixed_sources = [A, B, V, G]
    # Групи альтернатив: тільки Д
    groups = [D_group]

    # 2 сценарії: S1 (з Д1) або S2 (з Д2)
    scenarios = scenario_totals(fixed_sources, groups, total_name_prefix="S")
    # Загальна агрегація сценаріїв:
    # S_overall: "оболонка" як max(S1..S4) по μ(x) (стандартне OR по сценаріях)

    print("Фіксовані джерела:")
    for s in fixed_sources:
        print(" -", describe(s))

    print("\nГрупи альтернатив (OR):")
    for g in groups:
        print(f" - {g.name}: " + ", ".join(describe(o) for o in g.options))

    print("\nСценарії сум:")
    for s_name, chosen, total in scenarios:
        chosen_names = ", ".join(t.name for t in chosen)
        print(f" - {s_name}: {chosen_names}")
        print(f"   {describe(total)}")

    # --- Сітка для графіків ---
    dx = 1.0
    x_min = 0.0
    max_support_right = max(total.support_right for _, _, total in scenarios)
    x_max = max(max_support_right + 200, 500)
    x = np.arange(x_min, x_max + dx, dx, dtype=float)

    # --- Побудова кривих для установ ---
    mu_A = mu_trapezoid_from_5tuple(x, A)
    mu_B = mu_trapezoid_from_5tuple(x, B)
    mu_V = mu_trapezoid_from_5tuple(x, V)
    mu_G = mu_trapezoid_from_5tuple(x, G)

    # Д як OR-група
    mu_D1 = mu_trapezoid_from_5tuple(x, D1)
    mu_D2 = mu_trapezoid_from_5tuple(x, D2)
    mu_D = mu_union(x, D_group.options)

    # --- μ(x) для сценаріїв і загальна оболонка ---
    mu_scenarios: list[tuple[str, np.ndarray]] = []
    for s_name, _, total in scenarios:
        mu_scenarios.append((s_name, mu_trapezoid_from_5tuple(x, total)))
    mu_overall = np.maximum.reduce([mu for _, mu in mu_scenarios]) if mu_scenarios else np.zeros_like(x)

    # "Найбільш можливі" та "неможливі" обсяги для загальної оболонки
    most_possible = get_intervals(x, mu_overall >= 0.6)
    impossible = get_intervals(x, mu_overall < 0.01)
    print(f"\nНайбільш можливі обсяги (ядро, для S_overall, μ>=0.6): {most_possible}")
    print(f"\nНеможливі обсяги (в межах {x_min:g}-{x_max:g}, для S_overall, μ<0.01): {impossible}")

    # --- Етап 1: A, Б, В, Г, Д1, Д2 — КОЖЕН ОКРЕМО ---
    plot_single(x, mu_A, "Установа A (μA)", "A", color="black")
    plot_single(x, mu_B, "Установа Б (μБ)", "Б", color="tab:blue")
    plot_single(x, mu_V, "Установа В (μВ)", "В", color="tab:green")
    plot_single(x, mu_G, "Установа Г (μГ)", "Г", color="tab:orange")
    plot_single(x, mu_D1, "Установа Д1 (μД1)", "Д1", color="tab:red")
    plot_single(x, mu_D2, "Установа Д2 (μД2)", "Д2", color="tab:purple")

    # --- Етап 2: Д1, Д2, Д=max(Д1,Д2) — ОКРЕМО ---
    plt.figure(figsize=(10, 4))
    plt.plot(x, mu_D1, "--", alpha=0.75, label="Д1")
    plt.plot(x, mu_D2, "--", alpha=0.75, label="Д2")
    plt.plot(x, mu_D, color="tab:red", linewidth=3, alpha=0.85, label="Д = max(Д1, Д2)")
    plt.fill_between(x, mu_D, color="tab:red", alpha=0.10)
    plt.title("Установа Д як OR-альтернатива: Д1 або Д2")
    plt.xlabel("Сума фінансування (у.о.)")
    plt.ylabel("μ(x)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    # --- Етап 3: фінальний графік — сценарії сум і загальна оболонка ---
    plt.figure(figsize=(12, 6))
    for s_name, mu in mu_scenarios:
        plt.plot(x, mu, "--", alpha=0.55, label=s_name)

    overall_label = "S_overall"
    if len(mu_scenarios) == 1:
        overall_label = f"S_overall = {mu_scenarios[0][0]}"
    else:
        overall_label = f"S_overall = max(S1..S{len(mu_scenarios)})"

    plt.plot(x, mu_overall, color="gray", linewidth=3, alpha=0.5, label=overall_label)
    plt.fill_between(x, mu_overall, color="gray", alpha=0.15)

    plt.title("4.1. Можливий обсяг фінансування (сценарії та оболонка)")
    plt.xlabel("Сума фінансування (у.о.)")
    plt.ylabel("Ступінь впевненості μ(x)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

