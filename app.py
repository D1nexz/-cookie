import os
import sys
import threading
import time

import ctypes

# Важно: вызываем ДО импорта pyautogui. Без этого Windows с масштабом 125–150%
# отдаёт скриншот в логическом разрешении, а модель обучалась на физических
# пикселях (~1919x1079) — из-за этого печенька не распознавалась.
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

import cv2
import numpy as np
import pyautogui
from ultralytics import YOLO

pyautogui.FAILSAFE = True  # рывок мыши в угол экрана — экстренная остановка

# =====================================================================
# ГЛАВНОЕ ПРАВИЛО ЭТОЙ ВЕРСИИ
#
# Распознавание и клики разведены по разным потокам, чтобы клики по
# печеньке не зависели от скорости модели:
#
#   ПОТОК-ДЕТЕКТОР  — непрерывно делает скриншот и прогоняет его через
#                     YOLO, складывает свежие детекции в общий «слот».
#   ПОТОК-КЛИКЕР (main) — крутится очень часто (CHECK_INTERVAL) и:
#       1. ПОСТОЯННО кликает по печеньке (class 'game') — по свежим
#          позициям из слота, темпом GAME_CPS кликов в секунду;
#       2. ровно РАЗ В 5 СЕКУНД жмёт прокачку уровней (class 'upgrade
#          LvL') по точному таймере next_upgrade_time;
#       3. кликает РОВНО ОДИН РАЗ по золотой печеньке (class 'bonus').
#
#   Бонус и прокачку можно выключить через переменные окружения, чтобы
#   остались только клики по печеньке.
# =====================================================================


# =====================================================================
# Настройки (переопределяются переменными окружения)
# =====================================================================


def env_float(name, default):
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        print(f"ВНИМАНИЕ: {name}='{val}' — не число, использую {default}.")
        return default


def env_int(name, default):
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        print(f"ВНИМАНИЕ: {name}='{val}' — не число, использую {default}.")
        return default


def env_bool(name, default):
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ('1', 'true', 'yes', 'on', 'y', 'да')


# --- Что вообще делаем -------------------------------------------------
ENABLE_GAME = env_bool("ENABLE_GAME", True)          # клики по обычной печеньке
ENABLE_UPGRADES = env_bool("ENABLE_UPGRADES", True)  # нажатия на прокачку
ENABLE_BONUS = env_bool("ENABLE_BONUS", True)        # один клик по золотой печеньке

# Тестовый режим: ничего не кликает, только считает и печатает статистику.
DRY_RUN = env_bool("DRY_RUN", False)

# Ограничить время работы (сек). 0 — крутить вечно (Ctrl+C для остановки).
RUN_SECONDS = env_float("RUN_SECONDS", 0.0)

# Пороги уверенности по классам.
#   class 0 'game'        — обычная печенька: кликаем ПОСТОЯННО
#   class 1 'upgrade LvL' — кнопки прокачки: жмём раз в UPGRADE_INTERVAL секунд
#   class 2 'bonus'       — золотая печенька: кликаем РОВНО ОДИН РАЗ на появление
BONUS_CONFIDENCE = env_float("BONUS_CONF", 0.25)
GAME_CONFIDENCE = env_float("GAME_CONF", 0.25)
UPGRADE_CONFIDENCE = env_float("UPGRADE_CONF", 0.20)

# --- Шаг 1. Клики по печеньке (game) -----------------------------------
# Темп кликов, штуук в секунду (усреднённый, «токен-бакет»). 0 — без
# ограничения, кликать максимально быстро. GAME_CLICK_DELAY (сек между
# кликами) оставлен для совместимости: если задан, учитывается вместе.
GAME_CPS = env_float("GAME_CPS", 30.0)
GAME_CLICK_DELAY = env_float("GAME_DELAY", 0.0)
MAX_COOKIES_PER_TICK = env_int("MAX_COOKIES", 30)    # не более N кликов за виток
# Сколько секунд ещё кликаем по последним найденным печенькам, если
# детектор «моргнул» и свежий кадр ещё не готов. Больше — выше риск
# кликнуть мимо улетевшей печеньки.
GAME_MEMORY = env_float("GAME_MEMORY", 0.35)

# --- Шаг 2. Прокачка уровней (upgrade LvL) ------------------------------
# Прокачку жмём РОВНО раз в UPGRADE_INTERVAL секунд (по умолчанию 5).
UPGRADE_INTERVAL = env_float("UPGRADE_INTERVAL", 5.0)
UPGRADE_CLICK_DELAY = env_float("UPGRADE_DELAY", 0.05)   # пауза между кликами по прокачке
PAUSE_AFTER_UPGRADES = env_float("UPGRADE_PAUSE", 0.15)  # пауза после прокачки
# Модель на отдельных кадрах «моргает» и не видит кнопки прокачки. Чтобы
# из-за одного пропущенного кадра не потерять весь виток прокачки,
# запоминаем последние найденные позиции и жмём по ним.
UPGRADE_MEMORY = env_float("UPGRADE_MEMORY", 10.0)     # сек — сколько помним кнопки
UPGRADE_MATCH_RADIUS = env_int("UPGRADE_RADIUS", 40)   # px — считаем той же кнопкой

# --- Шаг 3. Золотая печенька (bonus) ------------------------------------
# Пока та же печенька видна на экране (или пропала короче чем на
# BONUS_GRACE секунд), повторных кликов по ней НЕ делаем.
BONUS_GRACE = env_float("BONUS_GRACE", 1.0)        # сек — запас на «мигание» детекции
BONUS_MATCH_RADIUS = env_int("BONUS_RADIUS", 60)   # px — в пределах считаем тем же объектом

# --- Циклы и фильтры детекций -------------------------------------------
# Виток потока-кликера: держим маленьким, чтобы клики шли непрерывно и
# таймер прокачки срабатывал точно (погрешность ~ CHECK_INTERVAL).
CHECK_INTERVAL = env_float("CHECK_INTERVAL", 0.005)
# Пауза между прогонами модели в потоке-детекторе. 0 — столько, сколько
# позволяет железо (обычно 10–30 кадров/с на CPU/GPU с yolov8n).
DETECT_INTERVAL = env_float("DETECT_INTERVAL", 0.0)
# Как часто печатать сводку (клики/с, кадров/с, число прокачек).
STATS_INTERVAL = env_float("STATS", 5.0)

# Кнопки прокачки — широкие горизонтальные прямоугольники (~14% ширины экрана).
# Мелкий мусор (иконки, курсор), который модель ошибочно помечает как
# "upgrade LvL", отсекаем по ширине рамки.
UPGRADE_MIN_WIDTH_RATIO = env_float("UPGRADE_MIN_W", 0.05)  # >= 5% ширины экрана

# Золотая печенька — заметный объект. Отсекаем мелкий мусор по площади рамки.
BONUS_MIN_AREA_RATIO = env_float("BONUS_MIN_AREA", 0.001)  # >= 0.1% площади экрана

# Обычная печенька: отсекаем и мусор (слишком мелкое), и ложные гигантские
# рамки (слишком крупное — обычно это фон/весь экран).
GAME_MIN_AREA_RATIO = env_float("GAME_MIN_AREA", 0.0002)  # >= 0.02% площади экрана
GAME_MAX_AREA_RATIO = env_float("GAME_MAX_AREA", 0.25)    # <= 25% площади экрана

SCREEN_AREA = 0
SCREEN_WIDTH = 0
SCREEN_HEIGHT = 0


# =====================================================================
# Поиск ЛУЧШЕЙ обученной модели (weights/best.pt)
# =====================================================================
DETECT_DIR = os.path.join('runs', 'detect')

# Подсказки для распознавания классов по их именам (регистр не важен).
CLASS_HINTS = {
    'game': ('game', 'cookie', 'печеньк'),
    'upgrade': ('upgrade', 'lvl', 'level', 'прокач'),
    'bonus': ('bonus', 'золот', 'gold'),
}


def _run_number(name):
    suffix = name.rsplit('-', 1)[-1] if '-' in name else '0'
    return int(suffix) if suffix.isdigit() else 0


def _read_final_map50(run_dir):
    """Последний mAP50(B) из results.csv запуска (0.0, если данных нет)."""
    csv_path = os.path.join(run_dir, 'results.csv')
    if not os.path.isfile(csv_path):
        return 0.0
    last = ''
    try:
        with open(csv_path, encoding='utf-8', errors='ignore') as fh:
            for line in fh:
                if line.strip():
                    last = line
    except OSError:
        return 0.0
    # столбцы: epoch,time,box,cls,dfl,precision,recall,mAP50,mAP50-95,...
    try:
        return float(last.strip().split(',')[7])
    except (IndexError, ValueError):
        return 0.0


def _class_hits(names):
    """Какие из нужных классов есть в модели: {'game': bool, 'upgrade': bool, ...}."""
    values = names.values() if hasattr(names, 'values') else names
    lowered = [str(n).lower() for n in values]
    return {
        key: any(any(hint in name for hint in hints) for name in lowered)
        for key, hints in CLASS_HINTS.items()
    }


def find_latest_model():
    """Выбирает лучшую модель, которая реально умеет находить печеньку.

    Приоритет:
      1) есть ВСЕ 3 класса (game + upgrade LvL + bonus);
      2) иначе — есть хотя бы класс 'upgrade LvL';
      3) среди подходящих — наибольший mAP50;
      4) при равенстве — наибольший номер запуска.
    Модель можно задать вручную переменной окружения MODEL_PATH.
    """
    override = os.environ.get('MODEL_PATH')
    if override:
        if os.path.isfile(override):
            print(f"Использую модель из MODEL_PATH: {override}")
            return override
        print(f"ВНИМАНИЕ: MODEL_PATH='{override}' не найден — ищу автоматически.")

    candidates = []
    if os.path.isdir(DETECT_DIR):
        for folder in os.listdir(DETECT_DIR):
            run_dir = os.path.join(DETECT_DIR, folder)
            weights_path = os.path.join(run_dir, 'weights', 'best.pt')
            if os.path.isfile(weights_path):
                candidates.append((folder, run_dir, weights_path))

    if not candidates:
        raise FileNotFoundError(
            "Не найдена обученная модель (best.pt). Сначала запустите train.py "
            "и дождитесь завершения обучения."
        )

    rows = []
    for folder, run_dir, weights_path in candidates:
        try:
            names = YOLO(weights_path).names
        except Exception as exc:  # noqa: BLE001 — модель может быть битой
            print(f"Пропускаю {folder}: не удалось прочитать модель ({exc})")
            continue
        hits = _class_hits(names)
        map50 = _read_final_map50(run_dir)
        has_all = all(hits.get(k) for k in ('game', 'upgrade', 'bonus'))
        key = (1 if has_all else 0,
               1 if hits.get('upgrade') else 0,
               map50,
               _run_number(folder))
        rows.append((key, folder, weights_path, hits, map50))

    if not rows:
        raise FileNotFoundError(
            "Не найдено ни одной читаемой модели (best.pt). Переобучите: train.py"
        )

    print("Кандидаты моделей (классы game/upgrade/bonus, mAP50):")
    for _key, folder, _weights, hits, map50 in sorted(rows, key=lambda r: r[0],
                                                     reverse=True):
        marks = ''.join('+' if hits.get(k) else '-'
                        for k in ('game', 'upgrade', 'bonus'))
        print(f"  {folder:22s} [{marks}] mAP50={map50:.3f}")

    rows.sort(key=lambda r: r[0], reverse=True)
    _key, folder, weights_path, hits, _map50 = rows[0]

    if not hits.get('game'):
        print("ВНИМАНИЕ: в модели нет класса 'game' (обычная печенька) — "
              "кликать по печеньке будет нечем. Проверьте data.yaml и переобучите.")
    if not hits.get('upgrade'):
        print("ВНИМАНИЕ: ни в одной модели нет класса 'upgrade LvL' — "
              "кнопки прокачки найдены НЕ будут.")
    if not hits.get('bonus'):
        print("ВНИМАНИЕ: ни в одной модели нет класса 'bonus' — "
              "золотая печенька найдена НЕ будет.")
    return weights_path


model_path = find_latest_model()
print(f"Загружаю модель: {model_path}")
model = YOLO(model_path)

# =====================================================================
# Определяем классы модели:  data.yaml: nc: 3, names: ['game', 'upgrade LvL', 'bonus']
#   класс 0 — обычная печенька (game)       — кликаем ПОСТОЯННО
#   класс 1 — кнопки прокачки (upgrade LvL) — жмём каждые UPGRADE_INTERVAL сек
#   класс 2 — золотая печенька (bonus)      — кликаем ОДИН РАЗ на появление
# =====================================================================
names = model.names

BONUS_CLASSES = []
UPGRADE_CLASSES = []
GAME_CLASSES = []
recognized_any = False
for idx, name in names.items():
    lower = str(name).lower()
    if any(k in lower for k in CLASS_HINTS['bonus']):
        BONUS_CLASSES.append(idx)
        recognized_any = True
    elif any(k in lower for k in CLASS_HINTS['upgrade']):
        UPGRADE_CLASSES.append(idx)
        recognized_any = True
    elif any(k in lower for k in CLASS_HINTS['game']):
        GAME_CLASSES.append(idx)
        recognized_any = True

# Если имена классов вообще не распознались — доверяем порядку из data.yaml.
if not recognized_any:
    if len(names) >= 3:
        GAME_CLASSES, UPGRADE_CLASSES, BONUS_CLASSES = [0], [1], [2]
    elif len(names) == 2:
        GAME_CLASSES, UPGRADE_CLASSES = [0], [1]
    else:
        GAME_CLASSES = [0]

print(f"Класс(ы) обычной печеньки (game): {GAME_CLASSES}")
print(f"Класс(ы) прокачки (upgrade LvL): {UPGRADE_CLASSES}")
print(f"Класс(ы) золотой печеньки (bonus): {BONUS_CLASSES}")


def init_screen():
    """Один скриншот ДО старта потоков: задаёт физические размеры экрана."""
    global SCREEN_AREA, SCREEN_WIDTH, SCREEN_HEIGHT
    screenshot = pyautogui.screenshot()
    SCREEN_WIDTH = screenshot.width
    SCREEN_HEIGHT = screenshot.height
    SCREEN_AREA = SCREEN_WIDTH * SCREEN_HEIGHT
    return np.ascontiguousarray(cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR))


def take_screenshot():
    """Скриншот экрана, BGR-массив, как ждёт YOLO."""
    screenshot = pyautogui.screenshot()
    frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    return np.ascontiguousarray(frame)


def click_xy(x, y):
    """Один клик мыши в координатах экрана (в пределах экрана)."""
    if SCREEN_WIDTH > 0 and SCREEN_HEIGHT > 0:
        x = max(0, min(int(x), SCREEN_WIDTH - 1))
        y = max(0, min(int(y), SCREEN_HEIGHT - 1))
    if DRY_RUN:
        return
    pyautogui.click(x, y)


def detect_objects(frame):
    """Прогоняет кадр через модель и возвращает (bonuses, games, upgrades).

    Единый формат для всех списков: (cx, cy, confidence, cls_id, area).
    bonuses  — золотая печенька (class 'bonus')
    games    — обычная печенька (class 'game')
    upgrades — кнопки прокачки  (class 'upgrade LvL')
    """
    # conf=0.05 — просим модель вернуть даже самые слабые детекции,
    # окончательные пороги применяем ниже по классам. verbose=False,
    # чтобы не спамить консоль YOLO-логом.
    results = model(frame, conf=0.05, verbose=False)

    bonuses = []
    games = []
    upgrades = []
    floor_conf = min(BONUS_CONFIDENCE, GAME_CONFIDENCE, UPGRADE_CONFIDENCE)

    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue

        for box in boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])

            if confidence < floor_conf:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            if not all(np.isfinite([x1, y1, x2, y2])):
                continue

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            width = x2 - x1
            height = y2 - y1
            area = width * height
            if area <= 0:
                continue

            if cls_id in GAME_CLASSES and confidence >= GAME_CONFIDENCE:
                # Отсекаем мусор и слишком крупные «рамки-на-весь-экран».
                if SCREEN_AREA > 0:
                    ratio = area / SCREEN_AREA
                    if ratio < GAME_MIN_AREA_RATIO or ratio > GAME_MAX_AREA_RATIO:
                        continue
                games.append((cx, cy, confidence, cls_id, area))

            elif cls_id in BONUS_CLASSES and confidence >= BONUS_CONFIDENCE:
                # Отсекаем мелкие случайные элементы (иконки, курсор), которые
                # модель иногда ошибочно помечает как "bonus".
                if SCREEN_AREA > 0 and area / SCREEN_AREA < BONUS_MIN_AREA_RATIO:
                    continue
                bonuses.append((cx, cy, confidence, cls_id, area))

            elif cls_id in UPGRADE_CLASSES and confidence >= UPGRADE_CONFIDENCE:
                # Отсекаем мелкие случайные элементы (иконки, курсор), которые
                # модель иногда ошибочно помечает как "upgrade LvL".
                if SCREEN_WIDTH > 0 and width / SCREEN_WIDTH < UPGRADE_MIN_WIDTH_RATIO:
                    continue
                upgrades.append((cx, cy, confidence, cls_id, area))

    return bonuses, games, upgrades


# =====================================================================
# ПОТОК-ДЕТЕКТОР: скриншот + инференс, результат — в общий слот.
# Благодаря ему клики по печеньке идут ровно, даже когда кадр считается
# долго (модель больше не «съедает» такт кликера).
# =====================================================================
class Detector(threading.Thread):
    def __init__(self):
        super().__init__(name="detector", daemon=True)
        self._lock = threading.Lock()
        self._snap = None                 # (ts, bonuses, games, upgrades)
        self._stop_evt = threading.Event()
        self.frames = 0
        self.errors = 0
        self.last_infer_ms = 0.0

    def run(self):
        while not self._stop_evt.is_set():
            started = time.monotonic()
            try:
                frame = take_screenshot()
                bonuses, games, upgrades = detect_objects(frame)
            except Exception as exc:  # noqa: BLE001 — поток обязан жить
                with self._lock:
                    self.errors += 1
                print(f"ДЕТЕКТОР: ошибка кадра ({exc}) — продолжаю.")
                time.sleep(0.25)
                continue
            elapsed = time.monotonic() - started
            with self._lock:
                self._snap = (time.monotonic(), bonuses, games, upgrades)
                self.frames += 1
                self.last_infer_ms = elapsed * 1000.0
            # Не грузим железо на 100%, если пауза задана.
            if DETECT_INTERVAL > 0:
                leftover = DETECT_INTERVAL - elapsed
                if leftover > 0:
                    time.sleep(leftover)

    def snapshot(self):
        with self._lock:
            return self._snap

    def stats(self):
        with self._lock:
            return self.frames, self.errors, self.last_infer_ms

    def stop(self):
        self._stop_evt.set()


# =====================================================================
# ШАГ 1. ОБЫЧНАЯ ПЕЧЕНЬКА (game): кликаем ПОСТОЯННО.
# =====================================================================
_game_clicks_total = 0
_game_bucket = 0.0          # «токен-бакет» разрешённых кликов
_game_bucket_t = 0.0


def game_click_budget(now, wanted):
    """Сколько кликов по печеньке разрешено сейчас (средний темп GAME_CPS).

    GAME_CPS <= 0 — без ограничения. Ёмкость бакета = GAME_CPS, т.е.
    доступен один секунд «запаса», чтобы печеньки, появившиеся после
    простоя, кликались сразу.
    """
    global _game_bucket, _game_bucket_t

    if GAME_CPS <= 0:
        return wanted

    # Стартуем с пустого бакета: темп выходит на заданный сразу, без
    # стартового «залпа» из целой секунды накопленных кликов.
    if _game_bucket_t == 0.0:
        _game_bucket_t = now
        _game_bucket = 0.0

    elapsed = max(0.0, now - _game_bucket_t)
    _game_bucket_t = now
    _game_bucket = min(GAME_CPS, _game_bucket + elapsed * GAME_CPS)

    # GAME_DELAY — дополнительная минимальная дистанция между кликами.
    if GAME_CLICK_DELAY > 0:
        _game_bucket = min(_game_bucket, max(0.0, int(elapsed / GAME_CLICK_DELAY)))

    allowed = int(min(wanted, _game_bucket))
    _game_bucket -= allowed
    return allowed


def click_games(games, now):
    """Кликает по обычным печенькам (class 'game') — ГЛАВНОЕ действие.

    Вызывается на КАЖДОМ витке потока-кликера, поэтому клики идут
    практически непрерывно. Возвращает число сделанных кликов.
    """
    global _game_clicks_total

    if not games:
        return 0

    # Самые «явные» печеньки — первыми; лишние (сверх MAX) отбрасываем.
    games = sorted(games, key=lambda g: -g[2])[:MAX_COOKIES_PER_TICK]
    allowed = game_click_budget(now, len(games))
    if allowed <= 0:
        return 0

    for cx, cy, _confidence, _cls_id, _area in games[:allowed]:
        click_xy(cx, cy)
        _game_clicks_total += 1
        if GAME_CLICK_DELAY > 0:
            time.sleep(GAME_CLICK_DELAY)

    return allowed


# =====================================================================
# ШАГ 2. ПРОКАЧКА УРОВНЕЙ (upgrade LvL): ровно раз в UPGRADE_INTERVAL сек.
# =====================================================================
# ——— Память о кнопках прокачки —————————————————————————————————————
# Каждая запись: {"cx", "cy", "confidence", "cls_id", "area", "last_seen"}.
_upgrade_memory = []
_upgrade_rounds = 0


def _same_upgrade_spot(cx, cy, spot):
    return max(abs(cx - spot["cx"]), abs(cy - spot["cy"])) <= UPGRADE_MATCH_RADIUS


def remember_upgrades(upgrades, now):
    """Обновляет память позиций кнопок прокачки по текущим детекциям."""
    global _upgrade_memory

    for cx, cy, confidence, cls_id, area in upgrades:
        updated = False
        for spot in _upgrade_memory:
            if _same_upgrade_spot(cx, cy, spot):
                spot.update(cx=cx, cy=cy, area=area,
                            confidence=confidence, cls_id=cls_id, last_seen=now)
                updated = True
                break
        if not updated:
            _upgrade_memory.append({
                "cx": cx, "cy": cy, "area": area,
                "confidence": confidence, "cls_id": cls_id, "last_seen": now,
            })

    # Забываем позиции, которых не видно дольше UPGRADE_MEMORY секунд.
    _upgrade_memory = [s for s in _upgrade_memory
                       if (now - s["last_seen"]) <= UPGRADE_MEMORY]


def perform_upgrades(upgrades, now=None):
    """Клик по всем кнопкам прокачки (снизу вверх) по списку детекций.

    Если детекций нет, но недавно (в пределах UPGRADE_MEMORY) кнопки уже
    находились — жмём по запомненным позициям.
    Возвращает True, если хотя бы одна кнопка прокачки была нажата.
    """
    global _upgrade_rounds

    if now is None:
        now = time.monotonic()

    remembered = None
    if not upgrades and _upgrade_memory:
        remembered = [(s["cx"], s["cy"], s["confidence"], s["cls_id"], s["area"])
                      for s in _upgrade_memory]
        upgrades = remembered

    if not upgrades:
        print("Прокачка: кнопки на экране не найдены — пропускаю.")
        return False

    # Сортируем по вертикальной координате по убыванию:
    # самая нижняя кнопка (обычно самая дорогая) кликается первой.
    upgrades = sorted(upgrades, key=lambda u: u[1], reverse=True)
    _upgrade_rounds += 1
    if remembered is not None:
        print(f"--- Прокачка #{_upgrade_rounds}: детекций нет, жму по {len(upgrades)} "
              f"недавно найденным кнопкам (память {UPGRADE_MEMORY:.0f} сек) ---")
    else:
        print(f"--- Прокачка #{_upgrade_rounds}: найдено {len(upgrades)} кнопок "
              f"(кликаю снизу вверх — от самой дорогой к самой дешёвой) ---")
    for cx, cy, confidence, cls_id, area in upgrades:
        click_xy(cx, cy)
        if UPGRADE_CLICK_DELAY > 0:
            time.sleep(UPGRADE_CLICK_DELAY)
        print(f"  Клик по прокачке ({cx}, {cy}), уверенность {confidence:.2f}, "
              f"класс {cls_id}, площадь {area:.0f}")
    if PAUSE_AFTER_UPGRADES > 0 and not DRY_RUN:
        time.sleep(PAUSE_AFTER_UPGRADES)
    return True


# =====================================================================
# ШАГ 3. ЗОЛОТАЯ ПЕЧЕНЬКА (bonus): РОВНО ОДИН клик на появление.
# =====================================================================
# Каждый трек — одна золотая печенька на экране:
#   {"cx", "cy"}  — последняя известная позиция
#   "clicked"     — по этой печенике уже кликнули (больше не жмём)
#   "last_seen"   — момент (monotonic), когда её видели последний раз
_bonus_tracks = []
_bonus_clicks = 0


def update_bonus_tracks(bonuses, now):
    """Кликает по НОВЫМ золотым печенькам — РОВНО ОДИН РАЗ на появление."""
    global _bonus_tracks, _bonus_clicks

    taken = [False] * len(_bonus_tracks)
    clicked = False

    # Самые уверенные детекции сопоставляем с треками первыми.
    for cx, cy, confidence, cls_id, area in sorted(bonuses, key=lambda b: -b[2]):
        best_i, best_d = None, None
        for i, track in enumerate(_bonus_tracks):
            if taken[i]:
                continue
            d = max(abs(cx - track["cx"]), abs(cy - track["cy"]))
            if d <= BONUS_MATCH_RADIUS and (best_d is None or d < best_d):
                best_i, best_d = i, d

        if best_i is None:
            track = {"cx": cx, "cy": cy, "clicked": False, "last_seen": now}
            _bonus_tracks.append(track)
        else:
            taken[best_i] = True
            track = _bonus_tracks[best_i]
            track["cx"], track["cy"] = cx, cy
            track["last_seen"] = now

        if track["clicked"]:
            # По ЭТОЙ ЖЕ печенике уже кликали — повторно не жмём, пока она висит.
            continue

        click_xy(cx, cy)
        track["clicked"] = True
        _bonus_clicks += 1
        clicked = True
        print(f"Клик по золотой печеньке ({cx}, {cy}) conf={confidence:.2f} "
              f"cls={cls_id} — РОВНО ОДИН РАЗ на появление.")

    # Треки, которых не видно дольше BONUS_GRACE, забываем: их следующее
    # появление будет считаться новым (и снова кликнется ровно один раз).
    _bonus_tracks = [t for t in _bonus_tracks if (now - t["last_seen"]) <= BONUS_GRACE]

    return clicked


# =====================================================================
# Главный цикл = поток-кликер.
# =====================================================================
if __name__ == '__main__':
    print("Запуск кликера. Клики и распознавание идут в разных потоках:")
    print("  поток-детектор : скриншот + YOLO, складывает свежие позиции в общий слот;")
    print("  поток-кликер   :")
    print(f"    1) обычная печенька (game) — ПОСТОЯННО, "
          f"{'~%.0f кликов/сек' % GAME_CPS if GAME_CPS > 0 else 'без ограничения темпа'};")
    print(f"    2) прокачка уровней (upgrade LvL) — "
          f"{'каждые %.0f сек' % UPGRADE_INTERVAL if ENABLE_UPGRADES else 'ОТКЛЮЧЕНА'};")
    print(f"    3) золотая печенька (bonus) — "
          f"{'ОДИН клик на появление' if ENABLE_BONUS else 'ОТКЛЮЧЕНА'};")
    if DRY_RUN:
        print("    РЕЖИМ DRY_RUN=1: клики НЕ выполняются, только диагностика.")
    if RUN_SECONDS > 0:
        print(f"    Остановлюсь через {RUN_SECONDS:.0f} сек (RUN_SECONDS).")
    print("Нажмите Ctrl+C для остановки. Рывок мыши в угол экрана — экстренная остановка.")

    # ---- Самопроверка: покажем, что модель видит на текущем экране ----
    # Первый скриншот заодно фиксирует физические размеры экрана (до старта
    # потока-детектора, чтобы не было гонки за глобальные переменные).
    try:
        frame = init_screen()
        bonuses, games, upgrades = detect_objects(frame)
        print(f"Экран: {SCREEN_WIDTH}x{SCREEN_HEIGHT}. "
              f"Самопроверка: обычная печенька: {len(games)}, "
              f"прокачка: {len(upgrades)}, золотая печенька: {len(bonuses)}.")
        for c in games:
            pct = (c[4] / SCREEN_AREA * 100) if SCREEN_AREA > 0 else 0.0
            print(f"  Обычная печенька ({c[0]}, {c[1]}) conf={c[2]:.2f} cls={c[3]} "
                  f"площадь={pct:.2f}%")
        for u in upgrades:
            pct = (u[4] / SCREEN_AREA * 100) if SCREEN_AREA > 0 else 0.0
            print(f"  Прокачка ({u[0]}, {u[1]}) conf={u[2]:.2f} cls={u[3]} "
                  f"площадь={pct:.2f}%")
        for b in bonuses:
            pct = (b[4] / SCREEN_AREA * 100) if SCREEN_AREA > 0 else 0.0
            print(f"  Золотая печенька ({b[0]}, {b[1]}) conf={b[2]:.2f} cls={b[3]} "
                  f"площадь={pct:.2f}%")
    except Exception as exc:
        print(f"Самопроверка не удалась: {exc}")

    detector = Detector()
    detector.start()

    # Точный таймер прокачки: следующий клик по кнопкам уровней — ровно
    # через UPGRADE_INTERVAL секунд (по умолчанию 5).
    start_t = time.monotonic()
    next_upgrade_time = start_t + UPGRADE_INTERVAL
    stats_t = start_t
    stats_game = 0
    stats_frames = 0
    stop_at = start_t + RUN_SECONDS if RUN_SECONDS > 0 else None

    last_snap_t = 0.0          # время последнего обработанного кадра детектора

    try:
        while True:
            now = time.monotonic()

            snap = detector.snapshot()
            if snap is None:          # первый кадр ещё считается
                time.sleep(CHECK_INTERVAL)
                continue
            snap_t, bonuses, games, upgrades = snap
            fresh = snap_t != last_snap_t
            if fresh:
                last_snap_t = snap_t

            # ============ ШАГ 1: ПОСТОЯННЫЕ КЛИКИ ПО ПЕЧЕНЬКЕ (game) ---------
            #    Кликаем каждый виток цикла (CHECK_INTERVAL), не дожидаясь
            #    нового кадра от детектора, пока позиции печеньек «свежие»
            #    (не старше GAME_MEMORY секунд).
            if ENABLE_GAME and (now - snap_t) <= GAME_MEMORY:
                click_games(games, now)

            # ============ ШАГ 2: ПРОКАЧКА УРОВНЕЙ (ровно раз в 5 секунд) ======
            if ENABLE_UPGRADES:
                # Позиции кнопок запоминаем по КАЖДОМУ новому кадру: если
                # таймер «выстрелит» в момент, когда модель моргнула, прокачка
                # всё равно уйдёт по памяти.
                if fresh:
                    remember_upgrades(upgrades, snap_t)
                if now >= next_upgrade_time:
                    perform_upgrades(upgrades, now)
                    # Отсчитываем интервал от момента ПОСЛЕ прокачки, чтобы
                    # долгие паузы не «накапливали» пропуски.
                    next_upgrade_time = time.monotonic() + UPGRADE_INTERVAL

            # ============ ШАГ 3: ЗОЛОТАЯ ПЕЧЕНЬКА (ОДИН клик на появление) ===
            if ENABLE_BONUS and fresh:
                update_bonus_tracks(bonuses, snap_t)

            # ---- Сводка раз в STATS_INTERVAL секунд ----
            if STATS_INTERVAL > 0 and (now - stats_t) >= STATS_INTERVAL:
                dt = now - stats_t
                frames, errors, infer_ms = detector.stats()
                cps = (_game_clicks_total - stats_game) / dt if dt > 0 else 0.0
                fps = (frames - stats_frames) / dt if dt > 0 else 0.0
                next_in = max(0.0, next_upgrade_time - now)
                print(f"[стат] кликов/сек: {cps:.1f} (всего {_game_clicks_total}) | "
                      f"кадров/с: {fps:.1f} ({infer_ms:.0f} мс/кадр) | "
                      f"прокачек: {_upgrade_rounds}, следующая через {next_in:.1f} с | "
                      f"бонусов: {_bonus_clicks}")
                stats_t = now
                stats_game = _game_clicks_total
                stats_frames = frames

            if stop_at is not None and now >= stop_at:
                break

            time.sleep(CHECK_INTERVAL)
    except KeyboardInterrupt:
        pass
    except pyautogui.FailSafeException:
        print("\nЭКСТРЕННАЯ ОСТАНОВКА: мышь увезена в угол экрана.")
    finally:
        detector.stop()
        worked = max(1e-6, time.monotonic() - start_t)
        frames, errors, infer_ms = detector.stats()
        print(f"\nОстановлено за {worked:.1f} сек.")
        print(f"  Кликов по обычной печеньке (game): {_game_clicks_total} "
              f"(~{_game_clicks_total / worked:.1f}/сек)")
        print(f"  Витков прокачки (upgrade LvL): {_upgrade_rounds} "
              f"(интервал {UPGRADE_INTERVAL:.0f} сек, ожидалось ~{worked / UPGRADE_INTERVAL:.1f})")
        print(f"  Кликов по золотой печеньке (bonus): {_bonus_clicks}")
        print(f"  Кадров распознано: {frames} (~{frames / worked:.1f} к/с, "
              f"последний кадр {infer_ms:.0f} мс, ошибок {errors})")
        sys.exit(0)
