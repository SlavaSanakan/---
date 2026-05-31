import pygame, random, math

pygame.init()

WIDTH, HEIGHT = 1400, 850
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Симуляция байесовской системы балансировки")
clock = pygame.time.Clock()

font = pygame.font.SysFont(None, 26)
big_font = pygame.font.SysFont(None, 36)

CONFIG = {
    "matrix_learning_rate": 0.005,
    "smoothing_alpha": 0.08,
    "max_y_change": 0.35,
    "update_interval": 700
}


class BayesianDDA:
    def __init__(self, skill_range=(-3, 3), num_points=200):
        self.theta = [
            skill_range[0] + i * (skill_range[1] - skill_range[0]) / (num_points - 1)
            for i in range(num_points)
        ]

        self.posterior = [self.gaussian(x, 0, 1) for x in self.theta]
        self.normalize()

    def gaussian(self, x, mu, sigma):
        return math.exp(-0.5 * ((x - mu) / sigma) ** 2)

    def normalize(self):
        s = sum(self.posterior)
        self.posterior = [p / s for p in self.posterior]

    def likelihood(self, theta, difficulty, outcome):
        prob_win = 1 / (1 + math.exp(-(theta - difficulty)))
        return prob_win if outcome == 1 else 1 - prob_win

    def update(self, difficulty, outcome):
        self.posterior = [
            p * self.likelihood(t, difficulty, outcome)
            for p, t in zip(self.posterior, self.theta)
        ]
        self.normalize()

    def expected_skill(self):
        return sum(t * p for t, p in zip(self.theta, self.posterior))

    def variance(self):
        mean = self.expected_skill()
        return sum(((t - mean) ** 2) * p for t, p in zip(self.theta, self.posterior))

    def confidence(self):
        uncertainty = math.sqrt(self.variance())
        return max(0.25, min(1.0, 1 - uncertainty / 3))


class LearnableMatrix:
    def __init__(self, rows=3, cols=5):
        self.rows = rows
        self.cols = cols

        self.W = [
            [random.uniform(0.05, 0.15) for _ in range(cols)]
            for _ in range(rows)
        ]

    def multiply(self, X):
        Y = []

        for col in range(self.cols):
            value = 0

            for row in range(self.rows):
                value += X[row] * self.W[row][col]

            Y.append(value)

        return Y

    def learn(self, X, Y, error):
        lr = CONFIG["matrix_learning_rate"]

        for r in range(self.rows):
            for c in range(self.cols):
                self.W[r][c] += lr * error * X[r] * (0.3 + abs(Y[c]))
                self.W[r][c] = max(-0.6, min(0.6, self.W[r][c]))


class AdaptiveDifficultySystem:
    def __init__(self):
        self.bayes = BayesianDDA()
        self.matrix = LearnableMatrix()

        self.base_spawn_rate = 90
        self.base_hp_mult = 1.0
        self.base_speed_mult = 1.0
        self.base_shooter_chance = 0.3
        self.base_bullet_speed = 6

        self.target_accuracy = 0.35
        self.target_kills = 4
        self.target_hp_lost = 1.5

        self.last_params = self.get_default_params()
        self.smoothed_params = self.get_default_params()

    def get_default_params(self):
        return {
            "spawn_rate": self.base_spawn_rate,
            "enemy_hp_mult": self.base_hp_mult,
            "enemy_speed_mult": self.base_speed_mult,
            "shooter_chance": self.base_shooter_chance,
            "enemy_bullet_speed": self.base_bullet_speed,
            "difficulty_index": 0,
            "Y": [0, 0, 0, 0, 0]
        }

    def smooth_value(self, old, new):
        alpha = CONFIG["smoothing_alpha"]
        return old + (new - old) * alpha

    def metrics_to_X(self, metrics):
        accuracy = metrics["accuracy"]
        kills = metrics["kills"]
        hp_lost = metrics["hp_lost"]

        x_accuracy = (accuracy - self.target_accuracy) * 2
        x_kills = kills / self.target_kills - 1
        x_survival = (self.target_hp_lost - hp_lost) / self.target_hp_lost

        x_accuracy = max(-1, min(1, x_accuracy))
        x_kills = max(-1, min(1, x_kills))
        x_survival = max(-1, min(1, x_survival))

        return [x_accuracy, x_kills, x_survival]

    def get_params_from_X(self, X):
        raw_Y = self.matrix.multiply(X)

        confidence = self.bayes.confidence()
        limit = CONFIG["max_y_change"]

        Y = [
            max(-limit, min(limit, y * confidence))
            for y in raw_Y
        ]

        spawn_pressure = Y[0]
        hp_pressure = Y[1]
        speed_pressure = Y[2]
        shooter_pressure = Y[3]
        bullet_pressure = Y[4]

        raw_spawn_rate = int(self.base_spawn_rate * (1 - spawn_pressure))
        raw_spawn_rate = max(30, min(180, raw_spawn_rate))

        raw_enemy_hp_mult = self.base_hp_mult * (1 + hp_pressure)
        raw_enemy_hp_mult = max(0.5, min(2.5, raw_enemy_hp_mult))

        raw_enemy_speed_mult = self.base_speed_mult * (1 + speed_pressure)
        raw_enemy_speed_mult = max(0.5, min(2.0, raw_enemy_speed_mult))

        raw_shooter_chance = self.base_shooter_chance + shooter_pressure
        raw_shooter_chance = max(0.05, min(0.75, raw_shooter_chance))

        raw_enemy_bullet_speed = self.base_bullet_speed * (1 + bullet_pressure)
        raw_enemy_bullet_speed = max(3, min(11, raw_enemy_bullet_speed))

        raw_difficulty_index = (
            (90 - raw_spawn_rate) / 60 +
            (raw_enemy_hp_mult - 1) +
            (raw_enemy_speed_mult - 1) +
            (raw_shooter_chance - 0.3) +
            (raw_enemy_bullet_speed - 6) / 5
        )

        self.smoothed_params["spawn_rate"] = self.smooth_value(
            self.smoothed_params["spawn_rate"],
            raw_spawn_rate
        )

        self.smoothed_params["enemy_hp_mult"] = self.smooth_value(
            self.smoothed_params["enemy_hp_mult"],
            raw_enemy_hp_mult
        )

        self.smoothed_params["enemy_speed_mult"] = self.smooth_value(
            self.smoothed_params["enemy_speed_mult"],
            raw_enemy_speed_mult
        )

        self.smoothed_params["shooter_chance"] = self.smooth_value(
            self.smoothed_params["shooter_chance"],
            raw_shooter_chance
        )

        self.smoothed_params["enemy_bullet_speed"] = self.smooth_value(
            self.smoothed_params["enemy_bullet_speed"],
            raw_enemy_bullet_speed
        )

        self.smoothed_params["difficulty_index"] = self.smooth_value(
            self.smoothed_params["difficulty_index"],
            raw_difficulty_index
        )

        self.smoothed_params["Y"] = Y

        return self.smoothed_params

    def update(self, metrics):
        X = self.metrics_to_X(metrics)
        params = self.get_params_from_X(X)

        performance_error = sum(X) / len(X)
        outcome = 1 if performance_error >= 0 else 0

        self.bayes.update(self.bayes.expected_skill(), outcome)
        self.matrix.learn(X, params["Y"], performance_error)

        self.last_params = params

        return params, X, performance_error


class VirtualPlayer:
    def __init__(self):
        self.aim = 0.5
        self.kill_power = 0.5
        self.survival = 0.5

    def clamp(self):
        self.aim = max(0, min(1, self.aim))
        self.kill_power = max(0, min(1, self.kill_power))
        self.survival = max(0, min(1, self.survival))

    def simulate_interval(self, params):
        spawn_difficulty = max(0, (90 - params["spawn_rate"]) / 60)
        hp_difficulty = params["enemy_hp_mult"] - 1
        speed_difficulty = params["enemy_speed_mult"] - 1
        shooter_difficulty = params["shooter_chance"] - 0.3
        bullet_difficulty = (params["enemy_bullet_speed"] - 6) / 5

        accuracy = (
            self.aim
            - speed_difficulty * 0.12
            - bullet_difficulty * 0.08
            + random.uniform(-0.025, 0.025)
        )
        accuracy = max(0, min(1, accuracy))

        kills = (
            self.kill_power * 8
            + accuracy * 3
            + spawn_difficulty * 1.0
            - hp_difficulty * 2.0
            - speed_difficulty * 0.8
            + random.uniform(-0.5, 0.5)
        )
        kills = max(0, int(kills))

        hp_lost = (
            (1 - self.survival) * 3
            + spawn_difficulty * 1.0
            + speed_difficulty * 1.5
            + shooter_difficulty * 2.0
            + bullet_difficulty * 1.5
            + random.uniform(-0.5, 0.5)
        )
        hp_lost = max(0, min(5, int(round(hp_lost))))

        return {
            "accuracy": accuracy,
            "kills": kills,
            "hp_lost": hp_lost
        }


def draw_text(text, x, y, color=(230,230,230)):
    screen.blit(font.render(text, True, color), (x, y))


def draw_graph(title, history, x, y, w, h, min_val, max_val, color):
    pygame.draw.rect(screen, (45,45,45), (x, y, w, h), 1)
    draw_text(title, x, y - 24)

    if len(history) < 2:
        return

    points = []
    visible = history[-w:]

    for i, value in enumerate(visible):
        px = x + i
        normalized = (value - min_val) / (max_val - min_val)
        normalized = max(0, min(1, normalized))
        py = y + h - normalized * h
        points.append((px, py))

    pygame.draw.lines(screen, color, False, points, 2)


def draw_bar(label, value, x, y, color):
    draw_text(label, x, y)
    pygame.draw.rect(screen, (50,50,50), (x + 140, y, 200, 18))
    pygame.draw.rect(screen, color, (x + 140, y, int(value * 200), 18))
    draw_text(f"{value:.2f}", x + 350, y)


dda_system = AdaptiveDifficultySystem()
player = VirtualPlayer()

history = {
    "difficulty": [],
    "skill": [],
    "spawn": [],
    "hp": [],
    "speed": [],
    "shooter": [],
    "bullet": [],
    "accuracy": [],
    "kills": [],
    "hp_lost": []
}

last_metrics = {
    "accuracy": 0,
    "kills": 0,
    "hp_lost": 0
}

last_update = pygame.time.get_ticks()
paused = False

running = True

while running:
    clock.tick(60)
    now = pygame.time.get_ticks()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

            if event.key == pygame.K_SPACE:
                paused = not paused

            if event.key == pygame.K_r:
                dda_system = AdaptiveDifficultySystem()
                history = {k: [] for k in history}

    keys = pygame.key.get_pressed()

    if keys[pygame.K_q]:
        player.aim += 0.01
    if keys[pygame.K_a]:
        player.aim -= 0.01

    if keys[pygame.K_w]:
        player.kill_power += 0.01
    if keys[pygame.K_s]:
        player.kill_power -= 0.01

    if keys[pygame.K_e]:
        player.survival += 0.01
    if keys[pygame.K_d]:
        player.survival -= 0.01

    player.clamp()

    if not paused and now - last_update >= CONFIG["update_interval"]:
        params = dda_system.last_params
        last_metrics = player.simulate_interval(params)

        params, X, error = dda_system.update(last_metrics)

        history["difficulty"].append(params["difficulty_index"])
        history["skill"].append(dda_system.bayes.expected_skill())
        history["spawn"].append(params["spawn_rate"])
        history["hp"].append(params["enemy_hp_mult"])
        history["speed"].append(params["enemy_speed_mult"])
        history["shooter"].append(params["shooter_chance"])
        history["bullet"].append(params["enemy_bullet_speed"])

        history["accuracy"].append(last_metrics["accuracy"])
        history["kills"].append(last_metrics["kills"])
        history["hp_lost"].append(last_metrics["hp_lost"])

        for k in history:
            history[k] = history[k][-700:]

        last_update = now

    params = dda_system.last_params

    screen.fill((18,18,18))

    draw_text("Стабильная байесовская система балансировки", 20, 15, (255,255,255))

    draw_text("Управление:", 20, 55)
    draw_text("Q/A = точность +/-", 20, 80)
    draw_text("W/S = сила убийств +/-", 20, 105)
    draw_text("E/D = выживаемость +/-", 20, 130)
    draw_text("SPACE = пауза", 20, 155)
    draw_text("R = сброс", 20, 180)
    draw_text("ESC = выход", 20, 205)

    draw_bar("Точность", player.aim, 20, 250, (80,180,255))
    draw_bar("Сила убийств", player.kill_power, 20, 285, (255,220,80))
    draw_bar("Выживаемость", player.survival, 20, 320, (100,255,120))

    draw_text(f"Последняя точность: {last_metrics['accuracy']:.2f}", 20, 370)
    draw_text(f"Последние убийства: {last_metrics['kills']}", 20, 395)
    draw_text(f"Потеряно HP: {last_metrics['hp_lost']}", 20, 420)

    draw_text(f"Оценка навыка Байесом: {dda_system.bayes.expected_skill():.2f}", 20, 470)
    draw_text(f"Уверенность Байеса: {dda_system.bayes.confidence():.2f}", 20, 495)

    draw_text(f"Скорость спавна: {params['spawn_rate']:.1f}", 20, 545)
    draw_text(f"Множитель HP врагов: {params['enemy_hp_mult']:.2f}", 20, 570)
    draw_text(f"Множитель скорости врагов: {params['enemy_speed_mult']:.2f}", 20, 595)
    draw_text(f"Шанс стрелков: {params['shooter_chance']:.2f}", 20, 620)
    draw_text(f"Скорость пуль: {params['enemy_bullet_speed']:.2f}", 20, 645)

    if paused:
        screen.blit(big_font.render("ПАУЗА", True, (255,80,80)), (20, 700))

    graph_x = 500
    graph_w = 820
    graph_h = 85

    draw_graph("Общий индекс сложности", history["difficulty"], graph_x, 60, graph_w, graph_h, -2, 3, (255,255,255))
    draw_graph("Оценка навыка Байесом", history["skill"], graph_x, 180, graph_w, graph_h, -3, 3, (255,200,80))
    draw_graph("Скорость спавна", history["spawn"], graph_x, 300, graph_w, graph_h, 30, 180, (80,180,255))
    draw_graph("Множитель HP врагов", history["hp"], graph_x, 420, graph_w, graph_h, 0.5, 2.5, (255,120,120))
    draw_graph("Множитель скорости врагов", history["speed"], graph_x, 540, graph_w, graph_h, 0.5, 2.0, (120,255,120))
    draw_graph("Шанс стрелков", history["shooter"], graph_x, 660, graph_w, graph_h, 0.05, 0.75, (200,120,255))

    pygame.display.flip()

pygame.quit()