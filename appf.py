import cv2
import numpy as np
import pyautogui
import time
from datetime import datetime
import os
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
import threading

class AdaptiveMotionDetectorScreen:
    def __init__(self):
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True, varThreshold=50, history=500)
        self.min_contour_area = 5000
        self.last_screenshot_time = 0
        self.screenshot_interval = 3
        self.is_running = False
        self.frame_count = 0
        
        # Параметры для адаптивного отслеживания
        self.adaptive_tracking = True
        self.template_update_interval = 30  # Обновляем шаблон каждые 30 кадров
        self.template_update_counter = 0
        self.region_template = None
        self.search_margin = 50  # Область поиска вокруг текущей позиции
        self.tracking_confidence_threshold = 0.6
        self.max_region_drift = 100  # Максимальное смещение области за раз
        
        # Сохраняем оригинальную область
        self.original_capture_region = None
        self.capture_region = None
        
        # Инициализация параметров фильтрации
        self.aspect_min = 0.2
        self.aspect_max = 5.0
        self.extent_min = 0.3
        
        # Создаем папку для скриншотов
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        self.screenshots_dir = os.path.join(desktop_path, "Motion_Screenshots")
        if not os.path.exists(self.screenshots_dir):
            os.makedirs(self.screenshots_dir)
        
        self.setup_gui()
    
    def setup_gui(self):
        """Создает GUI для управления программой"""
        self.root = tk.Tk()
        self.root.title("Adaptive Motion Detector - Screen Capture")
        self.root.geometry("500x600")
        
        # Вкладки для организации настроек
        tab_control = ttk.Notebook(self.root)
        
        # Вкладка основных настроек
        main_tab = ttk.Frame(tab_control)
        tab_control.add(main_tab, text='Основные')
        
        # Вкладка адаптивных настроек
        adaptive_tab = ttk.Frame(tab_control)
        tab_control.add(adaptive_tab, text='Адаптивное отслеживание')
        
        # Вкладка расширенных настроек
        advanced_tab = ttk.Frame(tab_control)
        tab_control.add(advanced_tab, text='Расширенные')
        
        tab_control.pack(expand=1, fill="both")
        
        # Основные настройки
        tk.Button(main_tab, text="Настроить область захвата", 
                 command=self.setup_capture_region, 
                 bg="lightblue", height=2).pack(pady=10)
        
        # Кнопка запуска/остановки
        self.start_button = tk.Button(main_tab, text="Запустить детекцию", 
                                     command=self.toggle_detection,
                                     bg="lightgreen", height=2)
        self.start_button.pack(pady=10)
        
        # Статус
        self.status_label = tk.Label(main_tab, text="Не запущено", 
                                   font=("Arial", 12))
        self.status_label.pack(pady=10)
        
        # Информация о смещении области
        self.drift_label = tk.Label(main_tab, text="Смещение: 0, 0", 
                                  font=("Arial", 10))
        self.drift_label.pack(pady=5)
        
        # Счетчик скриншотов
        self.counter_label = tk.Label(main_tab, text="Скриншотов: 0", 
                                    font=("Arial", 10))
        self.counter_label.pack(pady=5)
        
        # Кнопка сброса области
        tk.Button(main_tab, text="Сбросить область к начальной", 
                 command=self.reset_region, bg="orange").pack(pady=5)
        
        # Кнопка выхода
        tk.Button(main_tab, text="Выход", command=self.root.quit, 
                 bg="lightcoral").pack(pady=10)
        
        # Настройки адаптивного отслеживания
        self.adaptive_var = tk.BooleanVar(value=self.adaptive_tracking)
        tk.Checkbutton(adaptive_tab, text="Включить адаптивное отслеживание области", 
                      variable=self.adaptive_var).pack(pady=10)
        
        tk.Label(adaptive_tab, text="Интервал обновления шаблона (кадры):").pack(pady=5)
        self.template_interval_var = tk.StringVar(value=str(self.template_update_interval))
        tk.Entry(adaptive_tab, textvariable=self.template_interval_var, width=10).pack(pady=5)
        
        tk.Label(adaptive_tab, text="Область поиска (пиксели):").pack(pady=5)
        self.search_margin_var = tk.StringVar(value=str(self.search_margin))
        tk.Entry(adaptive_tab, textvariable=self.search_margin_var, width=10).pack(pady=5)
        
        tk.Label(adaptive_tab, text="Порог уверенности отслеживания (0-1):").pack(pady=5)
        self.confidence_var = tk.StringVar(value=str(self.tracking_confidence_threshold))
        tk.Entry(adaptive_tab, textvariable=self.confidence_var, width=10).pack(pady=5)
        
        tk.Label(adaptive_tab, text="Максимальное смещение за раз:").pack(pady=5)
        self.max_drift_var = tk.StringVar(value=str(self.max_region_drift))
        tk.Entry(adaptive_tab, textvariable=self.max_drift_var, width=10).pack(pady=5)
        
        # Расширенные настройки
        tk.Label(advanced_tab, text="Чувствительность (площадь объекта):").pack(pady=5)
        self.sensitivity_var = tk.StringVar(value=str(self.min_contour_area))
        tk.Entry(advanced_tab, textvariable=self.sensitivity_var, width=10).pack(pady=5)
        
        tk.Label(advanced_tab, text="Фильтр по соотношению сторон (min-max):").pack(pady=5)
        self.aspect_min_var = tk.StringVar(value=str(self.aspect_min))
        self.aspect_max_var = tk.StringVar(value=str(self.aspect_max))
        aspect_frame = tk.Frame(advanced_tab)
        aspect_frame.pack()
        tk.Entry(aspect_frame, textvariable=self.aspect_min_var, width=5).pack(side=tk.LEFT)
        tk.Label(aspect_frame, text=" - ").pack(side=tk.LEFT)
        tk.Entry(aspect_frame, textvariable=self.aspect_max_var, width=5).pack(side=tk.LEFT)
        
        tk.Label(advanced_tab, text="Минимальный экстент (0-1):").pack(pady=5)
        self.extent_min_var = tk.StringVar(value=str(self.extent_min))
        tk.Entry(advanced_tab, textvariable=self.extent_min_var, width=5).pack(pady=5)
        
        tk.Button(adaptive_tab, text="Применить настройки", 
                 command=self.apply_adaptive_settings).pack(pady=10)
        
        tk.Button(advanced_tab, text="Применить настройки", 
                 command=self.apply_settings).pack(pady=10)
        
        # Инструкции
        instructions = """
Адаптивное отслеживание:
1. Настройте начальную область захвата
2. Включите адаптивное отслеживание
3. Область будет автоматически следовать за движением камеры
4. При необходимости можно сбросить к начальной позиции
        """
        tk.Label(adaptive_tab, text=instructions, justify=tk.LEFT, 
                font=("Arial", 8)).pack(pady=10)
    
    def setup_capture_region(self):
        """Позволяет пользователю выбрать область экрана для захвата"""
        messagebox.showinfo("Настройка области", 
                          "Сейчас откроется окно для выбора области.\n"
                          "Нажмите и перетащите мышью, чтобы выбрать область с камерой.")
        
        # Временно скрываем окно
        self.root.withdraw()
        time.sleep(1)
        
        # Делаем скриншот всего экрана
        screenshot = pyautogui.screenshot()
        screenshot_np = np.array(screenshot)
        screenshot_cv = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
        
        # Позволяем пользователю выбрать область
        region = cv2.selectROI("Выберите область камеры", 
                              screenshot_cv, fromCenter=False)
        cv2.destroyAllWindows()
        
        # Возвращаем окно
        self.root.deiconify()
        
        if region != (0, 0, 0, 0):
            self.original_capture_region = region
            self.capture_region = region
            self.region_template = None  # Сбрасываем шаблон
            self.status_label.config(text=f"Область настроена: {region}")
            messagebox.showinfo("Успешно", "Область захвата настроена!")
        else:
            messagebox.showwarning("Отмена", "Область не выбрана")
    
    def apply_adaptive_settings(self):
        """Применяет настройки адаптивного отслеживания"""
        try:
            self.adaptive_tracking = self.adaptive_var.get()
            self.template_update_interval = int(self.template_interval_var.get())
            self.search_margin = int(self.search_margin_var.get())
            self.tracking_confidence_threshold = float(self.confidence_var.get())
            self.max_region_drift = int(self.max_drift_var.get())
            
            messagebox.showinfo("Успешно", 
                              f"Настройки адаптивного отслеживания применены:\n"
                              f"Отслеживание: {'Вкл' if self.adaptive_tracking else 'Выкл'}\n"
                              f"Интервал обновления: {self.template_update_interval}\n"
                              f"Область поиска: ±{self.search_margin} пикс\n"
                              f"Порог уверенности: {self.tracking_confidence_threshold}")
        except ValueError:
            messagebox.showerror("Ошибка", "Введите корректные числовые значения")
    
    def apply_settings(self):
        """Применяет расширенные настройки"""
        try:
            self.min_contour_area = int(self.sensitivity_var.get())
            self.aspect_min = float(self.aspect_min_var.get())
            self.aspect_max = float(self.aspect_max_var.get())
            self.extent_min = float(self.extent_min_var.get())
            
            messagebox.showinfo("Успешно", "Расширенные настройки применены!")
        except ValueError:
            messagebox.showerror("Ошибка", "Введите корректные числовые значения")
    
    def reset_region(self):
        """Сбрасывает область захвата к начальной позиции"""
        if self.original_capture_region is not None:
            self.capture_region = self.original_capture_region
            self.region_template = None
            self.template_update_counter = 0
            self.update_drift_info(0, 0)
            messagebox.showinfo("Сброс", "Область сброшена к начальной позиции")
    
    def capture_screen_region(self):
        """Захватывает выбранную область экрана"""
        if self.capture_region is None:
            return None
        
        x, y, w, h = self.capture_region
        screenshot = pyautogui.screenshot(region=(x, y, w, h))
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame
    
    def update_region_template(self, frame):
        """Обновляет шаблон для отслеживания области"""
        # Используем центральную часть кадра как шаблон
        h, w = frame.shape[:2]
        center_h, center_w = h // 4, w // 4
        template = frame[center_h:h-center_h, center_w:w-center_w]
        
        # Конвертируем в градации серого для лучшего matching
        self.region_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    def track_region(self):
        """Отслеживает и корректирует положение области захвата"""
        if not self.adaptive_tracking or self.region_template is None:
            return
        
        # Делаем скриншот большей области для поиска
        x, y, w, h = self.capture_region
        search_x = max(0, x - self.search_margin)
        search_y = max(0, y - self.search_margin)
        search_w = w + 2 * self.search_margin
        search_h = h + 2 * self.search_margin
        
        try:
            search_screenshot = pyautogui.screenshot(region=(search_x, search_y, search_w, search_h))
            search_frame = np.array(search_screenshot)
            search_frame = cv2.cvtColor(search_frame, cv2.COLOR_RGB2BGR)
            search_gray = cv2.cvtColor(search_frame, cv2.COLOR_BGR2GRAY)
            
            # Ищем шаблон в области поиска
            result = cv2.matchTemplate(search_gray, self.region_template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            # Если найдено хорошее совпадение
            if max_val > self.tracking_confidence_threshold:
                # Вычисляем новые координаты
                template_h, template_w = self.region_template.shape
                new_x_offset = max_loc[0]
                new_y_offset = max_loc[1]
                
                # Переводим в глобальные координаты
                new_x = search_x + new_x_offset - w//4  # Корректируем на смещение шаблона
                new_y = search_y + new_y_offset - h//4
                
                # Ограничиваем максимальное смещение
                drift_x = new_x - x
                drift_y = new_y - y
                
                if abs(drift_x) <= self.max_region_drift and abs(drift_y) <= self.max_region_drift:
                    # Плавное смещение (используем только часть найденного смещения)
                    smooth_factor = 0.3
                    final_x = int(x + drift_x * smooth_factor)
                    final_y = int(y + drift_y * smooth_factor)
                    
                    self.capture_region = (final_x, final_y, w, h)
                    
                    # Обновляем информацию о смещении в GUI
                    total_drift_x = final_x - self.original_capture_region[0]
                    total_drift_y = final_y - self.original_capture_region[1]
                    self.root.after(0, self.update_drift_info, total_drift_x, total_drift_y)
                    
                    return True
        
        except Exception as e:
            print(f"Ошибка при отслеживании области: {e}")
        
        return False
    
    def update_drift_info(self, drift_x, drift_y):
        """Обновляет информацию о смещении области"""
        self.drift_label.config(text=f"Смещение: {drift_x:+d}, {drift_y:+d}")
    
    def detect_motion(self):
        """Основной цикл детекции движения с адаптивным отслеживанием"""
        while self.is_running:
            try:
                frame = self.capture_screen_region()
                if frame is None:
                    time.sleep(0.1)
                    continue
                
                self.frame_count += 1
                self.template_update_counter += 1
                
                # Обновляем шаблон для отслеживания
                if (self.adaptive_tracking and 
                    (self.region_template is None or 
                     self.template_update_counter >= self.template_update_interval)):
                    self.update_region_template(frame)
                    self.template_update_counter = 0
                
                # Выполняем отслеживание области
                if self.frame_count > 30 and self.frame_count % 5 == 0:  # Каждые 5 кадров
                    tracking_success = self.track_region()
                    if tracking_success:
                        # Получаем обновленный кадр после корректировки области
                        frame = self.capture_screen_region()
                        if frame is None:
                            continue
                
                # Применяем детекцию движения после стабилизации
                if self.frame_count > 30:
                    # Конвертируем в grayscale для более стабильной обработки
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    # Применяем Gaussian blur для уменьшения шума
                    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
                    
                    fg_mask = self.background_subtractor.apply(blurred)
                    
                    # Убираем шум
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
                    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
                    
                    # Находим контуры
                    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, 
                                                 cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Проверяем наличие крупного движения с дополнительными критериями
                    motion_detected = False
                    largest_area = 0
                    
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        
                        if area > self.min_contour_area:
                            # Вычисляем соотношение сторон bounding box
                            x, y, w, h = cv2.boundingRect(contour)
                            aspect_ratio = w / h if h > 0 else 0
                            
                            # Вычисляем экстент
                            rect_area = w * h
                            extent = float(area) / rect_area if rect_area > 0 else 0
                            
                            # Фильтруем по форме и экстенту
                            if (self.aspect_min < aspect_ratio < self.aspect_max and 
                                extent > self.extent_min):
                                motion_detected = True
                                largest_area = max(largest_area, area)
                    
                    # Делаем скриншот при обнаружении движения
                    current_time = time.time()
                    if (motion_detected and 
                        (current_time - self.last_screenshot_time) > self.screenshot_interval):
                        
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        screenshot_filename = f"motion_detected_{timestamp}.jpg"
                        screenshot_path = os.path.join(self.screenshots_dir, screenshot_filename)
                        
                        # Сохраняем полный скриншот области
                        cv2.imwrite(screenshot_path, frame)
                        
                        self.last_screenshot_time = current_time
                        
                        # Формируем сообщение о статусе
                        status_text = f"ДВИЖЕНИЕ! Объект: {int(largest_area)} пикс"
                        if self.adaptive_tracking:
                            status_text += " [Адаптивное отслеживание]"
                        
                        # Обновляем GUI в основном потоке
                        self.root.after(0, self.update_status, status_text)
                        print(f"Движение обнаружено! Размер объекта: {int(largest_area)} пикселей")
                        print(f"Скриншот сохранен: {screenshot_filename}")
                    
                    elif motion_detected:
                        status_text = "Движение (недавний скриншот)"
                        if self.adaptive_tracking:
                            status_text += " [Отслеживание]"
                        self.root.after(0, self.update_status, status_text)
                    else:
                        status_text = "Ожидание движения..."
                        if self.adaptive_tracking:
                            status_text += " [Отслеживание активно]"
                        self.root.after(0, self.update_status, status_text)
                
                else:
                    # Стабилизация фона
                    self.background_subtractor.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                    self.root.after(0, self.update_status, 
                                  f"Стабилизация... {30 - self.frame_count}")
                
                # Обновляем счетчик скриншотов
                try:
                    screenshot_count = len([f for f in os.listdir(self.screenshots_dir) 
                                          if f.endswith('.jpg')])
                    self.root.after(0, self.update_counter, screenshot_count)
                except:
                    pass
                
                time.sleep(0.1)  # Небольшая задержка
                
            except Exception as e:
                print(f"Ошибка в детекции: {e}")
                time.sleep(1)
    
    def update_status(self, text):
        """Обновляет статус в GUI"""
        self.status_label.config(text=text)
    
    def update_counter(self, count):
        """Обновляет счетчик скриншотов"""
        self.counter_label.config(text=f"Скриншотов: {count}")
    
    def toggle_detection(self):
        """Запускает или останавливает детекцию"""
        if not self.is_running:
            if self.capture_region is None:
                messagebox.showwarning("Ошибка", 
                                     "Сначала настройте область захвата!")
                return
            
            self.is_running = True
            self.frame_count = 0
            self.template_update_counter = 0
            self.start_button.config(text="Остановить детекцию", bg="lightcoral")
            
            # Запускаем детекцию в отдельном потоке
            self.detection_thread = threading.Thread(target=self.detect_motion)
            self.detection_thread.daemon = True
            self.detection_thread.start()
            
            print("Адаптивная детекция движения запущена!")
            print(f"Начальная область захвата: {self.capture_region}")
            print(f"Адаптивное отслеживание: {'ВКЛ' if self.adaptive_tracking else 'ВЫКЛ'}")
            print(f"Скриншоты сохраняются в: {self.screenshots_dir}")
            
        else:
            self.is_running = False
            self.start_button.config(text="Запустить детекцию", bg="lightgreen")
            self.status_label.config(text="Остановлено")
            print("Детекция движения остановлена")
    
    def run(self):
        """Запускает приложение"""
        print("Adaptive Motion Detector - Screen Capture")
        print("Папка для скриншотов:", self.screenshots_dir)
        print("Адаптивное отслеживание области: ВКЛ")
        self.root.mainloop()

if __name__ == "__main__":
    # Устанавливаем fail-safe для pyautogui
    pyautogui.FAILSAFE = True
    
    detector = AdaptiveMotionDetectorScreen()
    detector.run()