import numpy as np
import matplotlib.pyplot as plt


class BayesianDDA:
    def __init__(self, skill_range=(-3, 3), num_points=200):
        """
        skill_range — диапазон возможного навыка игрока (theta)
        num_points — сколько точек используем для дискретизации распределения
        """

        # Создаём сетку возможных значений навыка игрока (theta)
        # Например: [-3, -2.97, ..., 3]
        self.theta = np.linspace(skill_range[0], skill_range[1], num_points)

        # ===== АПРИОРНОЕ РАСПРЕДЕЛЕНИЕ =====
        # Предполагаем, что изначально игрок "средний" (mu=0)
        # и используем нормальное распределение
        self.prior = self.gaussian(self.theta, mu=0, sigma=1)

        # Нормализация (чтобы сумма вероятностей = 1)
        self.prior /= self.prior.sum()

        # В начале posterior = prior (нет данных)
        self.posterior = self.prior.copy()

    def gaussian(self, x, mu, sigma):
        """
        Вычисляет значения нормального распределения (без коэффициента нормировки)
        Это просто "форма колокола"
        """
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    def likelihood(self, theta, difficulty, outcome):
        """
        likelihood = P(результат | навык)

        theta — возможные значения навыка
        difficulty — сложность уровня
        outcome — результат (1 = победа, 0 = поражение)
        """

        # Логистическая функция:
        # вероятность победы растёт, если навык > сложность
        prob_win = 1 / (1 + np.exp(-(theta - difficulty)))

        # Если игрок выиграл → используем P(win)
        # Если проиграл → используем P(loss) = 1 - P(win)
        if outcome == 1:
            return prob_win
        else:
            return 1 - prob_win

    def update(self, difficulty, outcome):
        """
        Обновление распределения навыка по формуле Байеса
        """

        # Считаем likelihood для каждого возможного theta
        likelihood_vals = self.likelihood(self.theta, difficulty, outcome)

        # ===== ФОРМУЛА БАЙЕСА =====
        # posterior ∝ likelihood * prior
        self.posterior = likelihood_vals * self.posterior

        # Нормализация (иначе это не распределение)
        self.posterior /= self.posterior.sum()

    def expected_skill(self):
        """
        Возвращает математическое ожидание навыка:
        E[theta] = сумма(theta * P(theta))
        """

        return np.sum(self.theta * self.posterior)

    def choose_difficulty(self):
        """
        Выбираем сложность на основе текущей оценки навыка

        Самый простой вариант:
        сложность = ожидаемый навык игрока
        """

        return self.expected_skill()

    def plot(self):
        """
        Визуализация распределения навыка
        """

        plt.plot(self.theta, self.posterior)
        plt.title("Posterior skill distribution")
        plt.xlabel("Skill (theta)")
        plt.ylabel("Probability")
        plt.grid()
        plt.show()


# ===============================
# ===== СИМУЛЯЦИЯ ИГРОКА =======
# ===============================

# Создаём систему DDA
dda = BayesianDDA()

# Это "истинный" навык игрока (игра его НЕ знает)
true_skill = 1.0

# Запускаем несколько раундов
for i in range(20):

    # 1. Игра выбирает сложность
    difficulty = dda.choose_difficulty()

    # 2. Симулируем вероятность победы игрока
    # (как если бы это была настоящая игра)
    prob_win = 1 / (1 + np.exp(-(true_skill - difficulty)))

    # 3. Генерируем случайный исход (win/loss)
    outcome = np.random.rand() < prob_win

    # 4. Обновляем модель на основе результата
    dda.update(difficulty, int(outcome))

    # Лог для понимания процесса
    print(
        f"Step {i:02d} | "
        f"difficulty={difficulty:.2f} | "
        f"prob_win={prob_win:.2f} | "
        f"outcome={'WIN' if outcome else 'LOSS'} | "
        f"estimated_skill={dda.expected_skill():.2f}"
    )

# Показываем итоговое распределение
dda.plot()