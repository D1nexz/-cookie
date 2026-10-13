import os
import time
from datetime import datetime

import pyautogui

# Папка для сохранения скриншотов (рядом со скриптом)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(BASE_DIR, "screenshots")
os.makedirs(save_dir, exist_ok=True)

print("Начинаю делать скриншоты каждые 5 секунд...")
print("Нажмите Ctrl+C для остановки\n")

count = 0

try:
    while True:
        # Делаем скриншот
        screenshot = pyautogui.screenshot()

        # Формируем имя файла с датой и временем
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(save_dir, f"screenshot_{timestamp}.png")

        # Сохраняем
        screenshot.save(filename)
        count += 1
        print(f"[{count}] Сохранено: {filename}")

        # Ждём 5 секунд
        time.sleep(5)

except KeyboardInterrupt:
    print(f"\nОстановлено. Всего сделано скриншотов: {count}")