#!/usr/bin/env python3
"""
🎯 Упрощенный детектор расстояния
Определяет расстояние до источника звука на основе децибел
"""

import numpy as np
import sounddevice as sd
import json
import time
import math
from datetime import datetime
from collections import deque
import scipy.signal

class SimpleDistanceDetector:
    def __init__(self, calibration_file='distance_calibration.json'):
        print("🎯 === ПРОСТОЙ ДЕТЕКТОР РАССТОЯНИЯ ===")
        print("📏 Расчет расстояния по громкости звука")
        print("🔊 Анализ децибел без ИИ классификации")
        print("=" * 50)
        
        # Параметры
        self.SAMPLE_RATE = 44100
        self.WINDOW_SIZE = 0.5  # Секунды
        self.CHUNK_SIZE = int(self.SAMPLE_RATE * self.WINDOW_SIZE)
        self.NOISE_THRESHOLD = 0.001
        self.MIN_VOLUME = 0.005
        
        # Буферы
        self.volume_buffer = deque(maxlen=10)
        self.distance_buffer = deque(maxlen=5)
        self.db_buffer = deque(maxlen=3)
        
        # Загрузка калибровочных данных
        self.load_calibration_data(calibration_file)
        
        # Калибровка микрофона
        self.calibrate_microphone()
        
        # Состояние
        self.running = False
        self.environment_type = "indoor"
        
    def load_calibration_data(self, filename):
        """Загрузка калибровочных данных из JSON"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                self.calibration = json.load(f)
            print(f"✅ Калибровочные данные загружены из {filename}")
            
            # Создаем таблицу расстояний
            self.distance_table = []
            mapping = self.calibration['calibration_data']['distance_db_mapping']
            for distance_str, data in mapping.items():
                distance = float(distance_str)
                db = data['db']
                self.distance_table.append((db, distance, data['description']))
            
            # Сортируем по убыванию дБ
            self.distance_table.sort(key=lambda x: x[0], reverse=True)
            
        except Exception as e:
            print(f"⚠️ Ошибка загрузки калибровки: {e}")
            self.create_default_calibration()
    
    def create_default_calibration(self):
        """Создание калибровки по умолчанию"""
        self.distance_table = [
            (70, 0.3, "Очень близко"),
            (65, 0.5, "Близко"),
            (60, 1.0, "Рядом"),
            (55, 1.5, "Близко"),
            (50, 2.0, "Средне"),
            (45, 3.0, "Далеко"),
            (40, 4.0, "Далеко"),
            (35, 5.0, "Очень далеко"),
            (30, 7.0, "Крайне далеко"),
            (25, 10.0, "На пределе")
        ]
        print("📊 Используется калибровка по умолчанию")
    
    def calibrate_microphone(self):
        """Калибровка микрофона"""
        print("🎙️ Калибровка микрофона...")
        print("🔇 Создайте тишину на 3 секунды...")
        
        # Запись тишины
        silence_samples = []
        for i in range(3):
            print(f"   {3-i}...")
            time.sleep(1)
            audio = sd.rec(int(self.SAMPLE_RATE * 0.5), samplerate=self.SAMPLE_RATE, channels=1, dtype=np.float32)
            sd.wait()
            silence_samples.append(np.abs(audio.flatten()).mean())
        
        self.noise_floor = max(np.mean(silence_samples), 1e-8)
        print(f"✅ Уровень шума: {self.noise_floor:.6f}")
    
    def calculate_decibels(self, audio_data):
        """Расчет децибел из аудио данных"""
        # RMS (Root Mean Square)
        rms = np.sqrt(np.mean(audio_data**2))
        
        # Избегаем логарифм от нуля
        if rms < self.noise_floor:
            rms = self.noise_floor
            
        # Конвертация в децибелы
        db = 20 * np.log10(rms / self.noise_floor)
        
        # Нормализация (экспериментальная коррекция)
        db = max(0, db + 60)  # Базовая коррекция
        
        return db
    
    def interpolate_distance(self, db_level):
        """Интерполяция расстояния на основе дБ"""
        # Поиск ближайших значений в таблице
        for i, (table_db, table_dist, desc) in enumerate(self.distance_table):
            if db_level >= table_db:
                if i == 0:
                    return table_dist, desc
                
                # Линейная интерполяция между двумя точками
                prev_db, prev_dist, prev_desc = self.distance_table[i-1]
                
                # Пропорция между точками
                ratio = (db_level - table_db) / (prev_db - table_db)
                distance = table_dist + ratio * (prev_dist - table_dist)
                
                return distance, f"Между {prev_desc.lower()} и {desc.lower()}"
        
        # Если дБ меньше минимального в таблице
        return self.distance_table[-1][1], self.distance_table[-1][2]
    
    def smooth_values(self, db_level, distance):
        """Сглаживание показаний"""
        self.db_buffer.append(db_level)
        self.distance_buffer.append(distance)
        
        if len(self.db_buffer) >= 3:
            smoothed_db = np.mean(list(self.db_buffer))
            smoothed_distance = np.median(list(self.distance_buffer))
            return smoothed_db, smoothed_distance
        
        return db_level, distance
    
    def format_distance(self, distance):
        """Форматирование расстояния"""
        if distance < 1.0:
            return f"{distance*100:.0f}см"
        elif distance < 10.0:
            return f"{distance:.1f}м"
        else:
            return f"{distance:.0f}м"
    
    def get_distance_icon(self, distance):
        """Иконка для расстояния"""
        if distance < 0.5:
            return "🔥"  # Очень близко
        elif distance < 1.5:
            return "📍"  # Близко
        elif distance < 3.0:
            return "📌"  # Средне
        elif distance < 5.0:
            return "📡"  # Далеко
        else:
            return "🌫️"  # Очень далеко
    
    def create_volume_bar(self, volume):
        """Визуальный индикатор громкости"""
        bar_length = 25
        max_volume = 0.1  # Максимальная ожидаемая громкость
        filled = int((volume / max_volume) * bar_length)
        filled = min(filled, bar_length)
        
        bar = "🔊 |" + "█" * filled + "░" * (bar_length - filled) + f"| {volume:.3f}"
        return bar
    
    def create_distance_scale(self, distance):
        """Шкала расстояния"""
        scale_length = 40
        max_distance = 10.0
        position = int((distance / max_distance) * scale_length)
        position = min(position, scale_length)
        
        scale = "📏 |" + "░" * position + "●" + "░" * (scale_length - position) + f"| {self.format_distance(distance)}"
        return scale
    
    def get_detection_quality(self, volume, db_level):
        """Оценка качества детекции"""
        if volume > 0.02 and db_level > 40:
            return "ОТЛИЧНО", "🟢"
        elif volume > 0.01 and db_level > 30:
            return "ХОРОШО", "🟡" 
        elif volume > 0.005 and db_level > 20:
            return "СРЕДНЕ", "🟠"
        else:
            return "СЛАБО", "🔴"
    
    def audio_callback(self, indata, frames, time, status):
        """Callback для обработки аудио"""
        if status:
            print(f"⚠️ Статус аудио: {status}")
        
        # Копируем данные
        audio_data = indata[:, 0].copy()
        
        # Измеряем громкость
        volume = np.abs(audio_data).mean()
        
        # Проверка на тишину
        if volume < self.NOISE_THRESHOLD:
            return
            
        # Буферизация
        self.volume_buffer.append(volume)
        
        # Фильтрация - показываем только значимые звуки
        if volume < self.MIN_VOLUME:
            return
        
        try:
            # Расчет децибел
            db_level = self.calculate_decibels(audio_data)
            
            # Расчет расстояния
            distance, description = self.interpolate_distance(db_level)
            
            # Сглаживание
            smoothed_db, smoothed_distance = self.smooth_values(db_level, distance)
            
            # Качество детекции
            quality, quality_icon = self.get_detection_quality(volume, smoothed_db)
            
            # Отображение
            self.display_detection(volume, smoothed_db, smoothed_distance, description, quality, quality_icon)
            
        except Exception as e:
            print(f"⚠️ Ошибка обработки: {e}")
    
    def display_detection(self, volume, db_level, distance, description, quality, quality_icon):
        """Отображение результатов"""
        current_time = datetime.now().strftime("%H:%M:%S")
        distance_icon = self.get_distance_icon(distance)
        
        # Очистка и отображение
        print("\n" + "=" * 60)
        print(f"🎯 === АНАЛИЗ РАССТОЯНИЯ === {current_time}")
        
        # Индикаторы
        volume_bar = self.create_volume_bar(volume)
        distance_scale = self.create_distance_scale(distance)
        
        print(volume_bar)
        print(distance_scale)
        
        # Основная информация
        print(f"\n📢 Децибелы:      {db_level:.1f} дБ")
        print(f"📏 Расстояние:    {distance_icon} {self.format_distance(distance)}")
        print(f"📝 Описание:      {description}")
        print(f"{quality_icon} Качество:      {quality}")
        print(f"🏠 Окружение:     {self.environment_type.upper()}")
        
        print("=" * 60)
    
    def list_audio_devices(self):
        """Список доступных аудио устройств"""
        print("🎤 Доступные аудио устройства:")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"   {i}: {device['name']} ✅")
    
    def start_detection(self):
        """Запуск детекции"""
        print("\n🎯 === ПАРАМЕТРЫ ДЕТЕКТОРА ===")
        print(f"   📊 Частота дискретизации: {self.SAMPLE_RATE} Гц")
        print(f"   🔇 Порог шума: {self.NOISE_THRESHOLD}")
        print(f"   🎚️ Минимальная громкость: {self.MIN_VOLUME}")
        print(f"   🏠 Окружение: {self.environment_type}")
        print("=" * 50)
        
        self.list_audio_devices()
        
        print("\n🎯 Детектор готов к работе!")
        print("🗣️ Говорите или издавайте звуки...")
        print("📏 Расстояние будет рассчитано автоматически")
        print("(Ctrl+C для выхода)\n")
        
        self.running = True
        
        try:
            with sd.InputStream(
                samplerate=self.SAMPLE_RATE,
                channels=1,
                callback=self.audio_callback,
                blocksize=self.CHUNK_SIZE,
                dtype=np.float32
            ):
                while self.running:
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            print("\n🛑 Остановка детектора...")
            self.running = False
        except Exception as e:
            print(f"❌ Ошибка: {e}")
        finally:
            print("👋 До свидания!")

def main():
    try:
        detector = SimpleDistanceDetector()
        detector.start_detection()
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")

if __name__ == "__main__":
    main()