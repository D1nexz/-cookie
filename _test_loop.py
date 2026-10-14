"""Проверка логики кликера БЕЗ игры на экране.

Запуск:  python _test_loop.py

Что проверяет:
  1) клики по печеньке (game) идут ПОСТОЯННО с заданным темпом GAME_CPS;
  2) прокачка (upgrade LvL) срабатывает ровно раз в UPGRADE_INTERVAL секунд;
  3) золотая печенька (bonus) кликается РОВНО ОДИН раз на появление.

Ничего не кликает по-настоящему: DRY_RUN=1, вместо скриншотов — подставные
детекции, вместо потока-детектора — фейк, выдающий заготовленные кадры.
"""
import os

os.environ["DRY_RUN"] = "1"
os.environ["GAME_CPS"] = "30"
os.environ["UPGRADE_INTERVAL"] = "5"
os.environ["STATS"] = "0"

import time

import app

CLICKS = []
app.click_xy = lambda x, y: CLICKS.append((round(time.monotonic(), 3), x, y))


class FakeDetector:
    """Возвращает всегда один и тот же кадр: 3 печеньки, 2 кнопки, 1 бонус."""

    def __init__(self):
        self.frames = 0
        self.errors = 0
        self.last_infer_ms = 5.0
        self._stop = False

    def start(self):
        pass

    def stop(self):
        self._stop = True

    def snapshot(self):
        self.frames += 1
        now = time.monotonic()
        bonuses = [(900, 300, 0.90, 2, 40000)]
        games = [(300, 400, 0.90, 0, 9000),
                 (360, 420, 0.80, 0, 9000),
                 (420, 440, 0.70, 0, 9000)]
        upgrades = [(1500, 800, 0.85, 1, 60000), (1500, 900, 0.75, 1, 60000)]
        return now, bonuses, games, upgrades

    def stats(self):
        return self.frames, self.errors, self.last_infer_ms


print("=" * 70)

# --- 1. Темп кликов по печеньке -----------------------------------------
CLICKS.clear()
app._game_clicks_total = 0
app._game_bucket = 0.0
app._game_bucket_t = 0.0
t0 = time.monotonic()
while time.monotonic() - t0 < 2.0:
    app.click_games([(300, 400, 0.9, 0, 9000)], time.monotonic())
    time.sleep(0.005)
cps = app._game_clicks_total / (time.monotonic() - t0)
print(f"1) Печенька (game) за {time.monotonic() - t0:.1f} сек: "
      f"{app._game_clicks_total} кликов (~{cps:.1f}/сек, цель ~30/сек)")
assert 20 <= cps <= 45, "темп кликов не соответствует GAME_CPS"

# --- 2. Прокачка раз в 5 секунд ------------------------------------------
CLICKS.clear()
app._upgrade_rounds = 0
app._upgrade_memory = []
interval = app.UPGRADE_INTERVAL
t0 = time.monotonic()
# Как в приложении: первый клик по прокачке — через UPGRADE_INTERVAL секунд.
next_upgrade = t0 + interval
fired = []
while time.monotonic() - t0 < 12.0:
    now = time.monotonic()
    if now >= next_upgrade:
        app.perform_upgrades([(1500, 800, 0.85, 1, 60000),
                              (1500, 900, 0.75, 1, 60000)], now)
        next_upgrade = time.monotonic() + interval
        fired.append(now - t0)
    time.sleep(0.005)
print(f"2) Прокачка (upgrade LvL) за 12 сек: {len(fired)} срабатывания "
      f"на секундах {[round(f, 1) for f in fired]}, кликов {len(CLICKS)}")
assert len(fired) == 2 and 4.5 < fired[0] < 5.5 and 9.5 < fired[1] < 10.5, \
    "прокачка должна сработать ровно на 5-й и 10-й секунде"
assert len(CLICKS) == 4, "за 2 витка прокачки должно быть 4 клика (2 кнопки x 2)"

# --- 3. Прокачка по памяти, когда модель «моргнула» -----------------------
CLICKS.clear()
now = time.monotonic()
app.remember_upgrades([(1500, 800, 0.85, 1, 60000),
                       (1500, 900, 0.75, 1, 60000)], now)
ok = app.perform_upgrades([], now)
print(f"3) Прокачка по памяти (детекций нет): нажала={ok}, кликов={len(CLICKS)}")
assert ok and len(CLICKS) == 2, "прокачка должна уйти по запомненным кнопкам"

# --- 4. Золотая печенька — ровно один клик на появление -------------------
CLICKS.clear()
app._bonus_tracks = []
app._bonus_clicks = 0
now = time.monotonic()
for _ in range(5):
    app.update_bonus_tracks([(900, 300, 0.9, 2, 40000)], now)
    now += 0.05
print(f"4) Бонус (bonus): 5 витков с одной и той же печенькой -> "
      f"кликов {app._bonus_clicks}")
assert app._bonus_clicks == 1, "по золотой печеньке кликаем РОВНО один раз"

# Новое появление (пропала с экрана дольше BONUS_GRACE) — кликается снова.
# Сначала витки БЕЗ бонуса: трек протухает и забывается.
now += app.BONUS_GRACE + 0.5
app.update_bonus_tracks([], now)
app.update_bonus_tracks([(900, 300, 0.9, 2, 40000)], now + 0.01)
print(f"   новое появление после {app.BONUS_GRACE:.1f} сек отсутствия -> "
      f"кликов всего {app._bonus_clicks}")
assert app._bonus_clicks == 2, "новое появление бонуса обязано кликнуться"

print("=" * 70)
print("ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ: game кликается постоянно, upgrade — раз в 5 сек,")
print("bonus — ровно один клик на появление.")
