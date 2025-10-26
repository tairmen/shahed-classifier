#!/usr/bin/env python3
"""
🎯 Android Sound Detector - Kivy Version
Универсальный детектор звуков для Android с определением расстояния
"""

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.progressbar import ProgressBar
from kivy.clock import Clock
from kivy.core.audio import SoundLoader
from kivy.utils import platform
from kivy.properties import StringProperty, NumericProperty, BooleanProperty

import numpy as np
import json
import os
from datetime import datetime

# Условный импорт в зависимости от платформы
if platform == 'android':
    from android.permissions import request_permissions, Permission
    request_permissions([Permission.RECORD_AUDIO, Permission.WRITE_EXTERNAL_STORAGE])
    # Для Android используем PyAudio или android.media
    try:
        from jnius import autoclass
        AudioRecord = autoclass('android.media.AudioRecord')
        AudioFormat = autoclass('android.media.AudioFormat')
        MediaRecorder = autoclass('android.media.MediaRecorder$AudioSource')
    except:
        print("⚠️ Не удалось импортировать Android audio API")
else:
    # Для тестирования на ПК
    try:
        import sounddevice as sd
    except:
        print("⚠️ sounddevice не установлен для тестирования")

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except:
    print("⚠️ TensorFlow не доступен - работа в режиме демо")
    TENSORFLOW_AVAILABLE = False


class SoundDetectorWidget(BoxLayout):
    """Главный виджет детектора звуков"""
    
    # Свойства для обновления UI
    status_text = StringProperty("Готов к работе")
    confidence_value = NumericProperty(0.0)
    distance_value = NumericProperty(0.0)
    db_value = NumericProperty(0.0)
    sound_type = StringProperty("---")
    detection_count = NumericProperty(0)
    is_detecting = BooleanProperty(False)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.padding = 20
        self.spacing = 15
        
        # Параметры детекции
        self.SAMPLE_RATE = 16000
        self.CHUNK_SIZE = int(self.SAMPLE_RATE * 0.975)
        self.CONFIDENCE_THRESHOLD = 0.6
        self.MIN_VOLUME = 0.002
        
        # Загрузка калибровки
        self.load_calibration()
        
        # Загрузка модели (если доступна)
        self.model_loaded = False
        if TENSORFLOW_AVAILABLE:
            self.load_model()
        
        # Построение UI
        self.build_ui()
        
        # Счетчики
        self.detection_count_value = 0
        self.frames_processed = 0
        
        # Аудио буфер
        self.audio_buffer = []
        
    def load_calibration(self):
        """Загрузка калибровочных данных"""
        try:
            calibration_file = 'distance_calibration.json'
            if os.path.exists(calibration_file):
                with open(calibration_file, 'r', encoding='utf-8') as f:
                    calibration = json.load(f)
                
                self.distance_table = []
                mapping = calibration['calibration_data']['distance_db_mapping']
                for distance_str, data in mapping.items():
                    distance = float(distance_str)
                    db = data['db']
                    self.distance_table.append((db, distance, data['description']))
                
                self.distance_table.sort(key=lambda x: x[0], reverse=True)
                print("✅ Калибровка загружена")
            else:
                self.create_default_calibration()
        except Exception as e:
            print(f"⚠️ Ошибка загрузки калибровки: {e}")
            self.create_default_calibration()
    
    def create_default_calibration(self):
        """Калибровка по умолчанию"""
        self.distance_table = [
            (70, 0.3, "Очень близко"),
            (65, 0.5, "Близко"),
            (60, 1.0, "Рядом"),
            (55, 1.5, "Недалеко"),
            (50, 2.0, "Средне"),
            (45, 3.0, "Далеко"),
            (40, 4.0, "Очень далеко"),
        ]
        print("📊 Используется калибровка по умолчанию")
    
    def load_model(self):
        """Загрузка AI модели"""
        try:
            model_path = 'model/my_sound_model.h5'
            if os.path.exists(model_path):
                self.classifier = tf.keras.models.load_model(model_path, compile=False)
                self.model_loaded = True
                print("✅ Модель загружена")
            else:
                print("⚠️ Модель не найдена")
                self.model_loaded = False
        except Exception as e:
            print(f"⚠️ Ошибка загрузки модели: {e}")
            self.model_loaded = False
    
    def build_ui(self):
        """Построение пользовательского интерфейса"""
        # Заголовок
        title = Label(
            text='🎵 Детектор Звуков',
            size_hint=(1, 0.1),
            font_size='24sp',
            bold=True,
            color=(0.2, 0.6, 1, 1)
        )
        self.add_widget(title)
        
        # Статус
        self.status_label = Label(
            text=self.status_text,
            size_hint=(1, 0.08),
            font_size='16sp',
            color=(1, 1, 1, 1)
        )
        self.add_widget(self.status_label)
        self.bind(status_text=lambda instance, value: setattr(self.status_label, 'text', value))
        
        # Информационная панель
        info_layout = BoxLayout(orientation='vertical', size_hint=(1, 0.35), spacing=10)
        
        # Уверенность
        conf_box = BoxLayout(orientation='horizontal', size_hint=(1, 0.2))
        conf_box.add_widget(Label(text='Уверенность:', size_hint=(0.4, 1), halign='left'))
        self.conf_label = Label(text='0%', size_hint=(0.3, 1), bold=True, color=(0, 1, 0, 1))
        conf_box.add_widget(self.conf_label)
        self.conf_bar = ProgressBar(max=100, size_hint=(0.3, 1))
        conf_box.add_widget(self.conf_bar)
        info_layout.add_widget(conf_box)
        
        # Расстояние
        dist_box = BoxLayout(orientation='horizontal', size_hint=(1, 0.2))
        dist_box.add_widget(Label(text='Расстояние:', size_hint=(0.4, 1), halign='left'))
        self.dist_label = Label(text='---', size_hint=(0.6, 1), bold=True, color=(1, 1, 0, 1))
        dist_box.add_widget(self.dist_label)
        info_layout.add_widget(dist_box)
        
        # Децибелы
        db_box = BoxLayout(orientation='horizontal', size_hint=(1, 0.2))
        db_box.add_widget(Label(text='Децибелы:', size_hint=(0.4, 1), halign='left'))
        self.db_label = Label(text='0 дБ', size_hint=(0.6, 1), bold=True, color=(1, 0.5, 0, 1))
        db_box.add_widget(self.db_label)
        info_layout.add_widget(db_box)
        
        # Тип звука
        type_box = BoxLayout(orientation='horizontal', size_hint=(1, 0.2))
        type_box.add_widget(Label(text='Тип звука:', size_hint=(0.4, 1), halign='left'))
        self.type_label = Label(text='---', size_hint=(0.6, 1), bold=True, color=(1, 0, 1, 1))
        type_box.add_widget(self.type_label)
        info_layout.add_widget(type_box)
        
        # Счетчик детекций
        count_box = BoxLayout(orientation='horizontal', size_hint=(1, 0.2))
        count_box.add_widget(Label(text='Детекций:', size_hint=(0.4, 1), halign='left'))
        self.count_label = Label(text='0', size_hint=(0.6, 1), bold=True, color=(0, 1, 1, 1))
        count_box.add_widget(self.count_label)
        info_layout.add_widget(count_box)
        
        self.add_widget(info_layout)
        
        # Кнопки управления
        button_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.15), spacing=10)
        
        self.start_button = Button(
            text='▶ СТАРТ',
            background_color=(0, 0.8, 0, 1),
            font_size='18sp',
            bold=True
        )
        self.start_button.bind(on_press=self.start_detection)
        button_layout.add_widget(self.start_button)
        
        self.stop_button = Button(
            text='⏸ СТОП',
            background_color=(0.8, 0, 0, 1),
            font_size='18sp',
            bold=True,
            disabled=True
        )
        self.stop_button.bind(on_press=self.stop_detection)
        button_layout.add_widget(self.stop_button)
        
        self.add_widget(button_layout)
        
        # Информация внизу
        info_text = Label(
            text=f'AI модель: {"✅ Загружена" if self.model_loaded else "❌ Не найдена"}\nРежим: {"Производительный" if TENSORFLOW_AVAILABLE else "Демо"}',
            size_hint=(1, 0.1),
            font_size='12sp',
            color=(0.7, 0.7, 0.7, 1)
        )
        self.add_widget(info_text)
    
    def start_detection(self, instance):
        """Запуск детекции"""
        self.is_detecting = True
        self.start_button.disabled = True
        self.stop_button.disabled = False
        self.status_text = "🔊 Детекция активна..."
        
        # Запускаем периодическую проверку аудио
        self.detection_event = Clock.schedule_interval(self.process_audio, 1.0)
        
        print("▶ Детекция запущена")
    
    def stop_detection(self, instance):
        """Остановка детекции"""
        self.is_detecting = False
        self.start_button.disabled = False
        self.stop_button.disabled = True
        self.status_text = "⏸ Детекция остановлена"
        
        # Останавливаем обработку
        if hasattr(self, 'detection_event'):
            self.detection_event.cancel()
        
        print("⏸ Детекция остановлена")
    
    def process_audio(self, dt):
        """Обработка аудио данных"""
        if not self.is_detecting:
            return
        
        # Симуляция обработки аудио (замените на реальный захват)
        # В реальном приложении здесь будет захват с микрофона
        
        # DEMO: Генерируем случайные данные для демонстрации
        if not self.model_loaded:
            self.simulate_detection()
        else:
            # Реальная обработка аудио
            self.real_audio_processing()
    
    def simulate_detection(self):
        """Симуляция детекции для демо режима"""
        import random
        
        # Случайная детекция с вероятностью 20%
        if random.random() < 0.2:
            self.detection_count_value += 1
            
            confidence = random.uniform(0.6, 1.0)
            db_level = random.uniform(50, 75)
            distance = self.interpolate_distance(db_level)[0]
            sound_types = ["речь/голос", "удар/стук", "музыка", "резкий звук"]
            sound_type = random.choice(sound_types)
            
            self.update_ui(confidence, db_level, distance, sound_type)
    
    def real_audio_processing(self):
        """Реальная обработка аудио с моделью"""
        # Здесь будет захват аудио с микрофона и обработка через модель
        # Для Android используйте AudioRecord API
        # Для ПК используйте sounddevice
        pass
    
    def interpolate_distance(self, db_level):
        """Интерполяция расстояния по децибелам"""
        for i, (table_db, table_dist, desc) in enumerate(self.distance_table):
            if db_level >= table_db:
                if i == 0:
                    return table_dist, desc
                
                prev_db, prev_dist, prev_desc = self.distance_table[i-1]
                ratio = (db_level - table_db) / (prev_db - table_db)
                distance = table_dist + ratio * (prev_dist - table_dist)
                return distance, desc
        
        return self.distance_table[-1][1], self.distance_table[-1][2]
    
    def format_distance(self, distance):
        """Форматирование расстояния"""
        if distance < 1.0:
            return f"{distance*100:.0f}см"
        elif distance < 10.0:
            return f"{distance:.1f}м"
        else:
            return f"{distance:.0f}м"
    
    def update_ui(self, confidence, db_level, distance, sound_type):
        """Обновление UI с результатами детекции"""
        # Обновляем свойства
        self.confidence_value = confidence * 100
        self.db_value = db_level
        self.distance_value = distance
        
        # Обновляем лейблы
        self.conf_label.text = f"{confidence*100:.1f}%"
        self.conf_bar.value = confidence * 100
        
        self.db_label.text = f"{db_level:.1f} дБ"
        self.dist_label.text = self.format_distance(distance)
        self.type_label.text = sound_type
        self.count_label.text = str(self.detection_count_value)
        
        # Меняем цвет в зависимости от уверенности
        if confidence > 0.8:
            self.conf_label.color = (0, 1, 0, 1)  # Зеленый
        elif confidence > 0.6:
            self.conf_label.color = (1, 1, 0, 1)  # Желтый
        else:
            self.conf_label.color = (1, 0.5, 0, 1)  # Оранжевый
        
        self.status_text = f"🟢 ЗВУК ОБНАРУЖЕН #{self.detection_count_value}"


class SoundDetectorApp(App):
    """Главное приложение"""
    
    def build(self):
        self.title = 'Sound Detector'
        return SoundDetectorWidget()
    
    def on_start(self):
        """Вызывается при старте приложения"""
        print("🎵 Sound Detector запущен")
        
        # Запрос разрешений для Android
        if platform == 'android':
            print("📱 Запрос разрешений Android...")
    
    def on_stop(self):
        """Вызывается при остановке приложения"""
        print("👋 Sound Detector остановлен")


if __name__ == '__main__':
    SoundDetectorApp().run()
