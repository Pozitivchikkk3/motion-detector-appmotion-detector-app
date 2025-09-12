# main.py для Kivy приложения
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.switch import Switch
from kivy.clock import Clock
from kivy.utils import platform

if platform == 'android':
    from android.permissions import request_permissions, Permission
    from android.storage import primary_external_storage_path
    import android
else:
    from unittest.mock import MagicMock
    android = MagicMock()

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.switch import Switch
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.image import Image

import cv2
import numpy as np
import os
from datetime import datetime

class MotionDetectorApp(App):
    def build(self):
        self.camera = None
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2()
        self.is_detecting = False
        
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        self.image = Image()
        layout.add_widget(self.image)
        
        self.status_label = Label(text='Готов к работе', size_hint_y=None, height=50)
        layout.add_widget(self.status_label)
        
        self.detection_switch = Switch(active=False)
        self.detection_switch.bind(active=self.toggle_detection)
        switch_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=50)
        switch_layout.add_widget(Label(text='Детекция движения:'))
        switch_layout.add_widget(self.detection_switch)
        layout.add_widget(switch_layout)
        
        return layout
    
    def toggle_detection(self, instance, value):
        if value:
            self.start_detection()
        else:
            self.stop_detection()
    
    def start_detection(self):
        try:
            self.camera = cv2.VideoCapture(0)
            self.is_detecting = True
            Clock.schedule_interval(self.process_frame, 1.0 / 30.0)  # 30 FPS
        except Exception as e:
            self.status_label.text = f'Ошибка: {str(e)}'
            self.detection_switch.active = False
    
    def stop_detection(self):
        self.is_detecting = False
        if self.camera:
            self.camera.release()
            self.camera = None
    
    def process_frame(self, dt):
        if not self.is_detecting or not self.camera:
            return
        
        ret, frame = self.camera.read()
        if ret:
            # Обработка кадра
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fg_mask = self.background_subtractor.apply(gray)
            
            # Конвертация для отображения в Kivy
            buf = cv2.flip(frame, 0).tostring()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.image.texture = texture

if __name__ == '__main__':
    MotionDetectorApp().run()
