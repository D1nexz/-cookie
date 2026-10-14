from ultralytics import YOLO

if __name__ == '__main__':

    # 1. Загрузка предобученной модели
    model = YOLO('yolov8n.pt')

    # 2. Запуск обучения
    model.train(
        data='data.yaml',
        epochs=100,        # количество эпох
        imgsz=640,         # размер изображения
        batch=4,           # маленький batch для маленького датасета (7 тренировочных картинок)
        workers=0,         # отключаем многопоточность (много проблем с ней)
        name='cookie_detector',
        patience=50        # остановка, если нет улучшений 50 эпох подряд
    )