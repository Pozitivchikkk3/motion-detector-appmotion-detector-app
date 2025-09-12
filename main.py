import cv2
import numpy as np
import os
from datetime import datetime
import time

def motion_detection_screenshot():
    # Пробуем разные способы инициализации камеры
    cap = None
    
    # Сначала пробуем DirectShow (обычно работает лучше на Windows)
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        print("Попытка подключения через DirectShow...")
    except:
        pass
    
    # Если DirectShow не работает, пробуем стандартный способ
    if cap is None or not cap.isOpened():
        try:
            cap = cv2.VideoCapture(0)
            print("Попытка подключения через стандартный драйвер...")
        except:
            pass
    
    # Проверяем, что камера работает
    if cap is None or not cap.isOpened():
        print("Ошибка: Не удается открыть камеру")
        print("Возможные причины:")
        print("1. Камера используется другим приложением")
        print("2. Нет разрешения на использование камеры")
        print("3. Драйверы камеры устарели")
        return
    
    # Устанавливаем параметры камеры
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Проверяем, что можем получить кадр
    ret, test_frame = cap.read()
    if not ret or test_frame is None:
        print("Ошибка: Не удается получить кадр с камеры")
        cap.release()
        return
    
    print(f"Камера инициализирована успешно. Разрешение: {test_frame.shape[1]}x{test_frame.shape[0]}")
    
    # Создаем папку для скриншотов на рабочем столе
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    screenshots_dir = os.path.join(desktop_path, "Motion_Screenshots")
    
    if not os.path.exists(screenshots_dir):
        os.makedirs(screenshots_dir)
        print(f"Создана папка: {screenshots_dir}")
    
    # Переменные для детекции движения
    background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    min_contour_area = 1000
    
    # Переменные для контроля частоты скриншотов
    last_screenshot_time = 0
    screenshot_interval = 2
    
    # Счетчик кадров для стабилизации фона
    frame_count = 0
    
    print("Детекция движения запущена. Нажмите 'q' для выхода.")
    print(f"Скриншоты сохраняются в: {screenshots_dir}")
    print("Подождите несколько секунд для стабилизации фона...")
    
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Ошибка чтения кадра, попытка переподключения...")
            time.sleep(1)
            continue
        
        frame_count += 1
        
        # Применяем фоновое вычитание только после стабилизации
        if frame_count > 30:  # Даем время для стабилизации фона
            fg_mask = background_subtractor.apply(frame)
            
            # Убираем шум
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            
            # Находим контуры
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Проверяем наличие значительного движения
            motion_detected = False
            for contour in contours:
                if cv2.contourArea(contour) > min_contour_area:
                    motion_detected = True
                    # Рисуем прямоугольник вокруг движущегося объекта
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Делаем скриншот при обнаружении движения
            current_time = time.time()
            if motion_detected and (current_time - last_screenshot_time) > screenshot_interval:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_filename = f"motion_detected_{timestamp}.jpg"
                screenshot_path = os.path.join(screenshots_dir, screenshot_filename)
                
                cv2.imwrite(screenshot_path, frame)
                print(f"Движение обнаружено! Скриншот сохранен: {screenshot_filename}")
                last_screenshot_time = current_time
            
            # Добавляем текст на изображение
            status_text = "ДВИЖЕНИЕ ОБНАРУЖЕНО!" if motion_detected else "Ожидание движения..."
            color = (0, 0, 255) if motion_detected else (0, 255, 0)
            
            # Показываем маску движения
            cv2.imshow('Motion Mask', fg_mask)
        else:
            # Пока идет стабилизация фона
            background_subtractor.apply(frame)
            status_text = f"Стабилизация фона... {30 - frame_count}"
            color = (255, 255, 0)
        
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Показываем количество сохраненных скриншотов
        try:
            screenshot_count = len([f for f in os.listdir(screenshots_dir) if f.endswith('.jpg')])
            cv2.putText(frame, f"Скриншотов: {screenshot_count}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        except:
            pass
        
        # Отображаем видео
        cv2.imshow('Motion Detection', frame)
        
        # Выход по нажатию 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Освобождаем ресурсы
    cap.release()
    cv2.destroyAllWindows()
    print("Детекция движения завершена.")

if __name__ == "__main__":
    motion_detection_screenshot()