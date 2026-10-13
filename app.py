import os
import sys
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
# Настройки кликов (можно переопределить через переменные окружения)
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


# Модель распознаёт печеньку (class 0 'game') слабо: даже на обучающих
# скриншотах уверенность 0.12-0.60 (проверено _diag.py), а на живом экране
# бывает 0.10-0.20. Поэтому порог снижен до 0.10, плюс в click_cookie()
# есть запасной путь по лучшей детекции.
# Кнопки прокачки (class 1 'upgrade LvL') модель видит уверенно: 0.65-0.99.
COOKIE_CONFIDENCE = env_float("COOKIE_CONF", 0.10)        # порог для печеньки (class 0)
UPGRADE_CONFIDENCE = env_float("UPGRADE_CONF", 0.30)      # порог для кнопок прокачки (class 1)

CLICKS_PER_COOKIE = env_int("COOKIE_CLICKS", 20)          # кликов по печеньке за подход
COOKIE_CLICK_DELAY = env_float("COOKIE_DELAY", 0.03)      # пауза между кликами по печеньке, сек
UPGRADE_CLICK_DELAY = env_float("UPGRADE_DELAY", 0.35)    # пауза между кликами по прокачке, сек
PAUSE_AFTER_UPGRADES = env_float("UPGRADE_PAUSE", 0.6)    # пауза после прокачки, сек

# Прокачка запускается раз в UPGRADE_INTERVAL секунд (по умолчанию — каждые 3 сек).
UPGRADE_INTERVAL = env_float("UPGRADE_INTERVAL", 3.0)
# Пауза между « подходами » к печеньке, сек.
CHECK_INTERVAL = env_float("CHECK_INTERVAL", 0.1)

# Кнопки прокачки — широкие горизонтальные прямоугольники (~14% ширины экрана).
# Мелкий мусор (иконки, курсор), который модель ошибочно помечает как
# "upgrade LvL", отсекаем по ширине рамки.
UPGRADE_MIN_WIDTH_RATIO = env_float("UPGRADE_MIN_W", 0.05)  # >= 5% ширины экрана

# Печенька — крупный объект. Отсекаем мелкий мусор (иконки, курсор),
# который модель ошибочно помечает как "game", по площади рамки.
COOKIE_MIN_AREA_RATIO = env_float("COOKIE_MIN_AREA", 0.003)  # >= 0.3% площади экрана

SCREEN_AREA = None
SCREEN_WIDTH = None
SCREEN_HEIGHT = None


# =====================================================================
# Поиск самой свежей обученной модели (weights/best.pt)
# =====================================================================
DETECT_DIR = os.path.join('runs', 'detect')


def find_latest_model():
    """Находит последний runs/detect/*/weights/best.pt (наибольший номер)."""
    candidates = []
    if os.path.isdir(DETECT_DIR):
        for folder in os.listdir(DETECT_DIR):
            weights_path = os.path.join(DETECT_DIR, folder, 'weights', 'best.pt')
            if os.path.isfile(weights_path):
                candidates.append((folder, weights_path))

    if not candidates:
        raise FileNotFoundError(
            "Не найдена обученная модель (best.pt). Сначала запустите train.py "
            "и дождитесь завершения обучения."
        )

    def sort_key(item):
        name = item[0]
        suffix = name.rsplit('-', 1)[-1] if '-' in name else '0'
        return int(suffix) if suffix.isdigit() else 0

    candidates.sort(key=sort_key, reverse=True)
    return candidates[0][1]


model_path = find_latest_model()
print(f"Загружаю модель: {model_path}")
model = YOLO(model_path)

# =====================================================================
# Определяем классы модели:             data.yaml: nc: 2, names: ['game', 'upgrade LvL']
#   класс 0 — печенька (game)           класс 1 — кнопки прокачки (upgrade LvL)
# =====================================================================
names = model.names

COOKIE_CLASSES = []
UPGRADE_CLASSES = []
for idx, name in names.items():
    lower = str(name).lower()
    if any(k in lower for k in ('game', 'cookie', 'печень')):
        COOKIE_CLASSES.append(idx)
    if any(k in lower for k in ('upgrade', 'lvl', 'level', 'прокач')):
        UPGRADE_CLASSES.append(idx)

# Если имена не распознались — полагаемся на порядок классов из data.yaml
if not COOKIE_CLASSES and len(names) >= 1:
    COOKIE_CLASSES = [0]
if not UPGRADE_CLASSES and len(names) >= 2:
    UPGRADE_CLASSES = [1]

print(f"Класс(ы) печеньки: {COOKIE_CLASSES}")
print(f"Класс(ы) прокачки: {UPGRADE_CLASSES}")

if len(names) < 2:
    print(
        "ВНИМАНИЕ: модель обучена с одним классом ('game'). "
        "Отредактируйте data.yaml (nc: 2) и переобучите модель: "
        "кнопки прокачки отдельно не распознаются."
    )


def take_screenshot():
    """Скриншот экрана, BGR-массив, как ждёт YOLO."""
    global SCREEN_AREA, SCREEN_WIDTH, SCREEN_HEIGHT
    screenshot = pyautogui.screenshot()
    if SCREEN_AREA is None:
        SCREEN_AREA = screenshot.width * screenshot.height
        SCREEN_WIDTH = screenshot.width
        SCREEN_HEIGHT = screenshot.height
    frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    return np.ascontiguousarray(frame)


def click_xy(x, y):
    """Один клик мыши в координатах экрана."""
    pyautogui.click(x, y)


def detect_objects(frame, cookie_conf=None):
    """Прогоняет кадр через модель и возвращает (cookies, upgrades).

    cookies  — список (cx, cy, confidence, cls_id, area)
    upgrades — список (cx, cy, area, confidence, cls_id)
    """
    if cookie_conf is None:
        cookie_conf = COOKIE_CONFIDENCE

    # conf=0.05 — просим модель вернуть даже самые слабые детекции,
    # окончательные пороги применяем ниже по классам. verbose=False,
    # чтобы не спамить консоль YOLO-логом.
    results = model(frame, conf=0.05, verbose=False)

    cookies = []
    upgrades = []

    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue

        for box in boxes:
            cls_id = int(box.cls[0].item())
            confidence = box.conf[0].item()

            if confidence < min(cookie_conf, UPGRADE_CONFIDENCE):
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            if not all(np.isfinite([x1, y1, x2, y2])):
                continue

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            width = x2 - x1
            height = y2 - y1
            area = width * height

            if cls_id in COOKIE_CLASSES and confidence >= cookie_conf:
                # Отсекаем мелкие случайные элементы (иконки, курсор), которые
                # модель иногда ошибочно помечает как "game".
                if SCREEN_AREA and area / SCREEN_AREA < COOKIE_MIN_AREA_RATIO:
                    continue
                cookies.append((cx, cy, confidence, cls_id, area))

            elif cls_id in UPGRADE_CLASSES and confidence >= UPGRADE_CONFIDENCE:
                # Отсекаем мелкие случайные элементы (иконки, курсор), которые
                # модель иногда ошибочно помечает как "upgrade LvL".
                if SCREEN_WIDTH:
                    width_ratio = width / SCREEN_WIDTH
                    if width_ratio < UPGRADE_MIN_WIDTH_RATIO:
                        continue
                upgrades.append((cx, cy, area, confidence, cls_id))

    return cookies, upgrades


def click_cookie():
    """Один проход по печеньке: скриншот, поиск, пачка кликов.

    Кликаем только по САМОЙ УВЕРЕННОЙ детекции: модель часто возвращает
    дубли/ложные 'game' рамки, и кликать по всем — значит тратить клики
    впустую. Возвращает True, если по печеньке кликнули.
    """
    frame = take_screenshot()
    cookies, _ = detect_objects(frame)

    if not cookies:
        # Запасной путь: берём любую детекцию класса 'game' с минимальным
        # порогом — лучше кликнуть по ней, чем простаивать.
        low, _ = detect_objects(frame, cookie_conf=0.05)
        if low:
            best = max(low, key=lambda c: c[2])
            print(f"Печенька: беру лучшую детекцию ({best[0]}, {best[1]}) "
                  f"при conf={best[2]:.2f} (порог {COOKIE_CONFIDENCE:.2f}).")
            cookies = [best]
        else:
            print("Печенька не найдена: ни одной детекции 'game' (conf>=0.05). "
                  "Игра, возможно, свёрнута или окно изменило размер.")
            return False

    best = max(cookies, key=lambda c: c[2])
    cx, cy = best[0], best[1]
    for _ in range(CLICKS_PER_COOKIE):
        click_xy(cx, cy)
        time.sleep(COOKIE_CLICK_DELAY)
    print(f"Клик по печеньке ({cx}, {cy}) conf={best[2]:.2f} — {CLICKS_PER_COOKIE} раз.")
    return True


def perform_upgrades():
    """Свежий скриншот + клик по всем кнопкам прокачки (снизу вверх).

    Возвращает True, хотя бы одна кнопка прокачки была найдена и нажата.
    """
    frame = take_screenshot()
    _, upgrades = detect_objects(frame)

    if not upgrades:
        print("Прокачка: кнопки на экране не найдены — пропускаю.")
        return False

    # Сортируем по вертикальной координате по убыванию:
    # самая нижняя кнопка (обычно самая дорогая) кликается первой.
    upgrades.sort(key=lambda u: u[1], reverse=True)
    print(f"--- Прокачка: найдено {len(upgrades)} кнопок "
          f"(кликаю снизу вверх — от самой дорогой к самой дешёвой) ---")
    for cx, cy, area, confidence, cls_id in upgrades:
        click_xy(cx, cy)
        time.sleep(UPGRADE_CLICK_DELAY)
        print(f"  Клик по прокачке ({cx}, {cy}), уверенность {confidence:.2f}, "
              f"класс {cls_id}, площадь {area:.0f}")
    time.sleep(PAUSE_AFTER_UPGRADES)
    return True


if __name__ == '__main__':
    print(f"Запуск кликера: печенька — постоянно, прокачка уровней — каждые "
          f"{UPGRADE_INTERVAL:.0f} сек после печеньки.")
    print("Нажмите Ctrl+C для остановки. Рывок мыши в угол экрана — экстренная остановка.")

    # ---- Самопроверка: покажем, что модель видит на текущем экране ----
    try:
        frame = take_screenshot()
        cookies, upgrades = detect_objects(frame)
        print(f"Самопроверка: печенька: {len(cookies)}, прокачка: {len(upgrades)}.")
        for c in cookies:
            pct = (c[4] / SCREEN_AREA * 100) if SCREEN_AREA else 0.0
            print(f"  Печенька ({c[0]}, {c[1]}) conf={c[2]:.2f} cls={c[3]} "
                  f"площадь={pct:.2f}%")
        for u in upgrades:
            print(f"  Прокачка ({u[0]}, {u[1]}) conf={u[4]:.2f} cls={u[3]} "
                  f"area={u[2]:.0f}")
    except Exception as exc:
        print(f"Самопроверка не удалась: {exc}")

    try:
        # Таймер прокачки: следующее нажатие на кнопки прокачки — через
        # UPGRADE_INTERVAL секунд. Печеньку кликаем в каждом витке цикла.
        next_upgrade_time = time.monotonic() + UPGRADE_INTERVAL

        while True:
            # 1) ЖМЁМ НА ПЕЧЕНЬКУ (свежий кадр, пачка кликов).
            if not click_cookie():
                print("Печенька не найдена — пробую ещё раз.")

            # 2) ПОСЛЕ ПЕЧЕНЬКИ, раз в 3 секунды, ЖМЁМ ПРОКАЧКУ УРОВНЕЙ.
            now = time.monotonic()
            if now >= next_upgrade_time:
                perform_upgrades()
                next_upgrade_time = time.monotonic() + UPGRADE_INTERVAL

            time.sleep(CHECK_INTERVAL)
    except KeyboardInterrupt:
        print("\nОстановлено.")
        sys.exit(0)
