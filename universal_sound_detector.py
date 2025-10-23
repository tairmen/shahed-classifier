#!/usr/bin/env python3
"""
🎯 Универсальный детектор звуков с определением расстояния
Распознает любые звуки и показывает информацию только при POSITIVE детекции
"""

import numpy as np
import sounddevice as sd
import tensorflow as tf
import tensorflow_hub as hub
import json
import time
import math
import os
from datetime import datetime
from collections import deque
import scipy.signal

class UniversalSoundDetector:
    def __init__(self, calibration_file='distance_calibration.json'):
        print("🎵 === УНИВЕРСАЛЬНЫЙ ДЕТЕКТОР ЗВУКОВ ===")
        print("🎯 Определение расстояния до любого источника звука")
        print("✅ Показ информации только при POSITIVE детекции")
        print("=" * 50)
        
        # Параметры
        self.SAMPLE_RATE = 16000
        self.WINDOW_SIZE = 0.975  # Секунды
        self.CHUNK_SIZE = int(self.SAMPLE_RATE * self.WINDOW_SIZE)
        self.CONFIDENCE_THRESHOLD = 0.6  # Снижен с 0.8 до 0.6
        self.NOISE_THRESHOLD = 0.001     # Снижен с 0.003 до 0.001
        self.MIN_VOLUME = 0.002          # Снижен с 0.008 до 0.002
        self.SMOOTHING_FRAMES = 3
        
        # Буферы для сглаживания
        self.prediction_buffer = deque(maxlen=self.SMOOTHING_FRAMES)
        self.volume_buffer = deque(maxlen=10)
        self.distance_buffer = deque(maxlen=5)
        self.db_buffer = deque(maxlen=3)
        
        # Счетчики
        self.positive_detections = 0
        self.total_attempts = 0
        self.debug_counter = 0  # Для отладки
        
        # Загрузка калибровочных данных
        self.load_calibration_data(calibration_file)
        
        # Загрузка модели
        self.load_model()
        
        # Калибровка микрофона
        self.calibrate_microphone()
        
        # Состояние
        self.running = False
        self.last_positive_detection = None
        self.environment_type = "indoor"
        
    def load_calibration_data(self, filename):
        """Загрузка калибровочных данных из JSON"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                self.calibration = json.load(f)
            print(f"✅ Калибровочные данные загружены из {filename}")
            
            # Создаем таблицу расстояний из JSON
            self.distance_table = []
            mapping = self.calibration['calibration_data']['distance_db_mapping']
            for distance_str, data in mapping.items():
                distance = float(distance_str)
                db = data['db']
                self.distance_table.append((db, distance, data['description']))
            
            # Сортируем по убыванию дБ
            self.distance_table.sort(key=lambda x: x[0], reverse=True)
            
            # Отладочная информация
            print(f"🔧 Загружено {len(self.distance_table)} записей калибровки:")
            for i, (db, dist, desc) in enumerate(self.distance_table[:3]):
                print(f"   {i+1}. {db}дБ = {dist}м - {desc}")
            if len(self.distance_table) > 3:
                print(f"   ... и еще {len(self.distance_table)-3} записей")
            
        except Exception as e:
            print(f"⚠️ Ошибка загрузки калибровки: {e}")
            print(f"🔄 Переход на калибровку по умолчанию...")
            self.create_default_calibration()
    
    def create_default_calibration(self):
        """Создание калибровки по умолчанию для любых звуков (только если JSON недоступен)"""
        self.distance_table = [
            (70, 0.3, "Очень близко - громкий звук"),      # Соответствует JSON
            (65, 0.5, "Близко - средний звук"),            # Соответствует JSON
            (60, 1.0, "Рядом - обычный звук"),             # Соответствует JSON
            (55, 1.5, "Близко - тихий звук"),              # Соответствует JSON
            (50, 2.0, "Средне - звук на расстоянии"),      # Соответствует JSON
            (45, 3.0, "Далеко - приглушенный звук"),       # Соответствует JSON
            (40, 4.0, "Далеко - слабый звук"),             # Соответствует JSON
            (35, 5.0, "Очень далеко - едва слышный звук"), # Соответствует JSON
            (30, 7.0, "Крайне далеко - отдаленный звук"),  # Соответствует JSON
            (25, 10.0, "На пределе слышимости")            # Соответствует JSON
        ]
        print("📊 Используется калибровка по умолчанию (соответствует JSON файлу)")
    
    def load_model(self):
        """Загрузка AI модели"""
        print("🔄 Загрузка YAMNet (локально)...")
        try:
            # Пытаемся загрузить локальную модель
            yamnet_path = './yamnet_local'
            if os.path.exists(yamnet_path):
                self.yamnet_model = tf.saved_model.load(yamnet_path)
                print("✅ YAMNet загружен локально из ./yamnet_local")
            else:
                print("⚠️ Локальная модель не найдена, загружаем из интернета...")
                self.yamnet_model = hub.load('https://www.kaggle.com/models/google/yamnet/frameworks/TensorFlow2/variations/yamnet/versions/1')
                print("✅ YAMNet загружен из интернета")
        except Exception as e:
            print(f"⚠️ Ошибка загрузки локальной модели: {e}")
            print("🔄 Попытка загрузки из интернета...")
            try:
                self.yamnet_model = hub.load('https://www.kaggle.com/models/google/yamnet/frameworks/TensorFlow2/variations/yamnet/versions/1')
                print("✅ YAMNet загружен из интернета (резервный способ)")
            except Exception as e2:
                print(f"❌ Не удалось загрузить YAMNet: {e2}")
                self.yamnet_model = None
        
        print("🔄 Загрузка классификатора (ваша обученная модель)...")
        try:
            # Пытаемся загрузить .h5 файл
            self.classifier = tf.keras.models.load_model('model/my_sound_model.h5', compile=False)
            print("✅ H5 модель загружена (обучена на ваших звуках)")
        except Exception as e:
            print(f"⚠️ Ошибка загрузки H5: {e}")
            try:
                # Пытаемся альтернативный путь
                self.classifier = tf.keras.models.load_model('model', compile=False)
                print("✅ SavedModel загружен (обучена на ваших звуках)")
            except Exception as e2:
                print(f"❌ Ошибка загрузки модели: {e2}")
                print("🤖 Работаем без AI классификации")
                self.classifier = None
    
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
        
        # Нормализация согласно JSON калибровке (70дБ = 30см)
        db = max(0, db + 60)  # Возвращаем к +60 для соответствия JSON
        
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
    
    def classify_audio(self, audio_data):
        """Классификация аудио с помощью AI"""
        if self.classifier is None or self.yamnet_model is None:
            return "unknown", 0.5
        
        try:
            # Ресемплинг для YAMNet (только если нужно)
            if len(audio_data) != self.CHUNK_SIZE:
                resampled_audio = scipy.signal.resample(audio_data, self.CHUNK_SIZE)
            else:
                resampled_audio = audio_data
            resampled_audio = np.float32(resampled_audio)
            
            # YAMNet эмбеддинги (кэшируемые)
            scores, embeddings, spectrogram = self.yamnet_model(resampled_audio)
            
            # Берем среднее по времени для стабильности
            mean_embedding = np.mean(embeddings.numpy(), axis=0)
            mean_embedding = np.expand_dims(mean_embedding, axis=0)
            
            # Ваша модель предсказание
            prediction = self.classifier.predict(mean_embedding, verbose=0)
            confidence = float(np.max(prediction))
            
            # Определяем класс по вашей модели
            predicted_class = "positive" if prediction[0][1] > prediction[0][0] else "negative"
            
            return predicted_class, confidence
            
        except Exception as e:
            print(f"⚠️ Ошибка классификации: {e}")
            return "error", 0.0
    
    def smooth_values(self, db_level, distance):
        """Сглаживание показаний"""
        self.db_buffer.append(db_level)
        self.distance_buffer.append(distance)
        
        if len(self.db_buffer) >= 3:
            smoothed_db = np.mean(list(self.db_buffer))
            smoothed_distance = np.median(list(self.distance_buffer))
            return smoothed_db, smoothed_distance
        
        return db_level, distance
    
    def determine_sound_type(self, audio_data, db_level):
        """Определение типа звука по частотам и характеристикам"""
        # FFT анализ
        fft = np.fft.fft(audio_data)
        magnitude = np.abs(fft[:len(fft)//2])
        
        # Частотные диапазоны
        low_end = int(300 * len(magnitude) / (self.SAMPLE_RATE/2))    # До 300 Гц
        mid_end = int(2000 * len(magnitude) / (self.SAMPLE_RATE/2))   # 300-2000 Гц
        high_end = int(8000 * len(magnitude) / (self.SAMPLE_RATE/2))  # 2000-8000 Гц
        
        low_energy = np.mean(magnitude[10:low_end])
        mid_energy = np.mean(magnitude[low_end:mid_end])
        high_energy = np.mean(magnitude[mid_end:high_end])
        
        total = low_energy + mid_energy + high_energy
        if total < 1e-6:
            return "тишина", "🔇"
        
        low_ratio = low_energy / total
        mid_ratio = mid_energy / total
        high_ratio = high_energy / total
        
        # Классификация по частотному составу и громкости (согласно JSON)
        if mid_ratio > 0.6 and high_ratio < 0.3:
            return "речь/голос", "🗣️"
        elif low_ratio > 0.5 and db_level > 70:  # Возвращено к JSON значению
            return "удар/стук", "🥁"
        elif high_ratio > 0.4:
            return "резкий звук", "⚡"
        elif mid_ratio > 0.4 and high_ratio > 0.2:
            return "музыка", "🎵"
        elif low_ratio > 0.6:
            return "низкий гул", "🌊"
        elif db_level > 75:  # Возвращено к значению выше JSON максимума
            return "громкий звук", "📢"
        else:
            return "смешанный звук", "🔊"
    
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
    
    def create_progress_bars(self, volume, distance, confidence):
        """Создание индикаторов"""
        # Громкость
        vol_bar_len = 25
        vol_filled = int((min(volume, 0.1) / 0.1) * vol_bar_len)
        vol_bar = f"🔊 |{'█' * vol_filled}{'░' * (vol_bar_len - vol_filled)}| {volume:.3f}"
        
        # Расстояние
        dist_bar_len = 30
        dist_pos = int((min(distance, 10) / 10) * dist_bar_len)
        dist_bar = f"📏 |{'░' * dist_pos}●{'░' * (dist_bar_len - dist_pos)}| {self.format_distance(distance)}"
        
        # Уверенность
        conf_bar_len = 20
        conf_filled = int(confidence * conf_bar_len)
        conf_bar = f"🎯 |{'█' * conf_filled}{'░' * (conf_bar_len - conf_filled)}| {confidence:.2f}"
        
        return vol_bar, dist_bar, conf_bar
    
    def show_status_dot(self):
        """Показать точку активности (без спама в консоль)"""
        print(".", end="", flush=True)
    
    def audio_callback(self, indata, frames, time, status):
        """Callback для обработки аудио"""
        if status:
            print(f"⚠️ Статус аудио: {status}")
        
        # Копируем данные
        audio_data = indata[:, 0].copy()
        
        # Измеряем громкость
        volume = np.abs(audio_data).mean()
        
        # Увеличиваем счетчик попыток
        self.total_attempts += 1
        
        # Проверка на тишину
        if volume < self.MIN_VOLUME:
            # Показываем точку активности каждые 100 кадров (реже)
            if self.total_attempts % 100 == 0:
                self.show_status_dot()
            return
            
        try:
            # Классификация звука
            predicted_class, confidence = self.classify_audio(audio_data)
            
            # ОТЛАДКА: Показываем первые 5 детекций для понимания работы
            if self.debug_counter < 5 and volume > self.MIN_VOLUME:
                self.debug_counter += 1
                db_level = self.calculate_decibels(audio_data)
                print(f"\n🔧 ОТЛАДКА #{self.debug_counter}: volume={volume:.4f}, db={db_level:.1f}, class={predicted_class}, conf={confidence:.3f}")
            
            # ПОКАЗЫВАЕМ ИНФОРМАЦИЮ ТОЛЬКО ДЛЯ POSITIVE!
            if predicted_class == "positive" and confidence >= self.CONFIDENCE_THRESHOLD:
                # Расчет децибел
                db_level = self.calculate_decibels(audio_data)
                
                # Расчет расстояния
                distance, description = self.interpolate_distance(db_level)
                
                # Сглаживание
                smoothed_db, smoothed_distance = self.smooth_values(db_level, distance)
                
                # Определение типа звука
                sound_type, sound_icon = self.determine_sound_type(audio_data, smoothed_db)
                
                # Увеличиваем счетчик положительных детекций
                self.positive_detections += 1
                
                # Отображение результатов
                self.display_positive_detection(
                    volume, smoothed_db, smoothed_distance, description,
                    confidence, sound_type, sound_icon
                )
            else:
                # Для negative или низкой уверенности - только точка
                if self.total_attempts % 50 == 0:  # Увеличен с 20 до 50
                    self.show_status_dot()
            
        except Exception as e:
            print(f"⚠️ Ошибка обработки: {e}")
    
    def display_positive_detection(self, volume, db_level, distance, description, 
                                 confidence, sound_type, sound_icon):
        """Отображение только положительных детекций"""
        current_time = datetime.now().strftime("%H:%M:%S")
        distance_icon = self.get_distance_icon(distance)
        
        # Создание индикаторов
        vol_bar, dist_bar, conf_bar = self.create_progress_bars(volume, distance, confidence)
        
        # Очистка линии точек и новая строка
        print("\n")
        
        # Заголовок
        print(f"🟢 === ЗВУК ОБНАРУЖЕН #{self.positive_detections} === {current_time}")
        
        # Индикаторы
        print(vol_bar)
        print(dist_bar)
        print(conf_bar)
        
        # Основная информация
        print(f"\n🎯 Статус:        ✅ POSITIVE (целевой звук)")
        print(f"📊 Уверенность:   {confidence:.2f} ({confidence*100:.1f}%)")
        print(f"📢 Децибелы:      {db_level:.1f} дБ")
        print(f"📏 Расстояние:    {distance_icon} {self.format_distance(distance)}")
        print(f"📝 Описание:      {description}")
        print(f"{sound_icon} Тип звука:     {sound_type.upper()}")
        print(f"🏠 Окружение:     {self.environment_type.upper()}")
        print(f"📈 Статистика:    {self.positive_detections} детекций из {self.total_attempts} попыток")
        
        print("=" * 70)
        
        # Сохранение
        self.last_positive_detection = {
            'confidence': confidence,
            'distance': distance,
            'db_level': db_level,
            'sound_type': sound_type,
            'time': current_time
        }
    
    def list_audio_devices(self):
        """Список доступных аудио устройств"""
        print("🎤 Доступные аудио устройства:")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"   {i}: {device['name']} ✅")
    
    def start_detection(self):
        """Запуск детекции"""
        print("\n🎯 === ПАРАМЕТРЫ УНИВЕРСАЛЬНОГО ДЕТЕКТОРА ===")
        print(f"   📊 Частота дискретизации: {self.SAMPLE_RATE} Гц")
        print(f"   🎚️ Минимальная уверенность: {self.CONFIDENCE_THRESHOLD}")
        print(f"   🔇 Порог шума: {self.NOISE_THRESHOLD}")
        print(f"   🎚️ Минимальная громкость: {self.MIN_VOLUME}")
        print(f"   🏠 Окружение: {self.environment_type}")
        print(f"   🤖 AI классификация: {'✅' if self.classifier else '❌'}")
        print(f"   🧠 YAMNet (feature extractor): {'✅' if self.yamnet_model else '❌'}")
        print(f"   🎯 Цель обнаружения: ВАШИ СПЕЦИФИЧЕСКИЕ ЗВУКИ")
        print(f"   📊 Модель: Обучена на ваших данных")
        print("=" * 50)
        
        self.list_audio_devices()
        
        print("\n🎵 Универсальный детектор готов к работе!")
        print("🔊 Издавайте любые звуки - голос, музыку, стуки...")
        print("✅ Информация показывается только при обнаружении ЦЕЛЕВЫХ звуков")
        print("🔧 ОТЛАДКА: Для демонстрации будут показаны первые 5 детекций любого уровня")
        print("⏱️ Активность детектора: ", end="", flush=True)
        
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
            print(f"\n\n🛑 Остановка детектора...")
            print(f"📊 Всего обнаружено целевых звуков: {self.positive_detections}")
            print(f"📈 Эффективность: {self.positive_detections}/{self.total_attempts} = {(self.positive_detections/max(self.total_attempts,1)*100):.1f}%")
            self.running = False
        except Exception as e:
            print(f"❌ Ошибка: {e}")
        finally:
            print("👋 До свидания!")

def main():
    try:
        detector = UniversalSoundDetector()
        detector.start_detection()
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")

if __name__ == "__main__":
    main()