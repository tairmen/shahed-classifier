#!/usr/bin/env python3
"""
🎯 Простой универсальный детектор звуков
Обнаруживает любые звуки без AI, показывает только значимые детекции
"""

import numpy as np
import sounddevice as sd
import json
import time
import math
from datetime import datetime
from collections import deque

class SimpleUniversalSoundDetector:
    def __init__(self, calibration_file='distance_calibration.json'):
        print("🎵 === ПРОСТОЙ УНИВЕРСАЛЬНЫЙ ДЕТЕКТОР ===")
        print("🎯 Обнаружение любых звуков без AI")
        print("✅ Показ только значимых звуков")
        print("=" * 50)
        
        # Параметры
        self.SAMPLE_RATE = 44100
        self.WINDOW_SIZE = 0.5  # Секунды
        self.CHUNK_SIZE = int(self.SAMPLE_RATE * self.WINDOW_SIZE)
        
        # Пороги для универсального детектора
        self.VOLUME_THRESHOLD = 0.015      # Минимальная громкость
        self.ENERGY_THRESHOLD = 0.008      # Энергетический порог
        self.PEAK_THRESHOLD = 0.05         # Порог для пиков
        self.FREQUENCY_THRESHOLD = 100     # Минимальная частота
        
        # Буферы
        self.volume_history = deque(maxlen=20)
        self.db_buffer = deque(maxlen=5)
        self.distance_buffer = deque(maxlen=5)
        
        # Счетчики
        self.detections_count = 0
        self.frames_processed = 0
        
        # Загрузка калибровочных данных
        self.load_calibration_data(calibration_file)
        
        # Калибровка
        self.calibrate_microphone()
        
        # Состояние
        self.running = False
        self.last_detection_time = 0
        self.min_interval = 0.3  # Минимальный интервал между детекциями
        
    def load_calibration_data(self, filename):
        """Загрузка калибровочных данных"""
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
        """Калибровка по умолчанию (соответствует JSON)"""
        self.distance_table = [
            (70, 0.3, "Очень близко - громкий звук"),
            (65, 0.5, "Близко - средний звук"),
            (60, 1.0, "Рядом - обычный звук"),
            (55, 1.5, "Близко - тихий звук"),
            (50, 2.0, "Средне - звук на расстоянии"),
            (45, 3.0, "Далеко - приглушенный звук"),
            (40, 4.0, "Далеко - слабый звук"),
            (35, 5.0, "Очень далеко - едва слышный звук"),
            (30, 7.0, "Крайне далеко - отдаленный звук"),
            (25, 10.0, "На пределе слышимости")
        ]
        print("📊 Используется калибровка по умолчанию (соответствует JSON файлу)")
    
    def calibrate_microphone(self):
        """Калибровка микрофона"""
        print("🎙️ Калибровка микрофона...")
        print("🔇 Создайте тишину на 3 секунды...")
        
        silence_samples = []
        for i in range(3):
            print(f"   {3-i}...")
            time.sleep(1)
            audio = sd.rec(int(self.SAMPLE_RATE * 0.5), samplerate=self.SAMPLE_RATE, channels=1, dtype=np.float32)
            sd.wait()
            silence_samples.append(np.abs(audio.flatten()).mean())
        
        self.noise_floor = max(np.mean(silence_samples), 1e-8)
        self.background_volume = max(np.mean(silence_samples), 1e-6)
        print(f"✅ Фоновый шум: {self.background_volume:.6f}")
        
        # Адаптивные пороги на основе фонового шума
        self.adaptive_volume_threshold = max(self.VOLUME_THRESHOLD, self.background_volume * 3)
        self.adaptive_energy_threshold = max(self.ENERGY_THRESHOLD, self.background_volume * 2)
        
        print(f"🎯 Адаптивный порог громкости: {self.adaptive_volume_threshold:.6f}")
        print(f"⚡ Адаптивный энергетический порог: {self.adaptive_energy_threshold:.6f}")
    
    def calculate_decibels(self, audio_data):
        """Расчет децибел"""
        rms = np.sqrt(np.mean(audio_data**2))
        if rms < self.noise_floor:
            rms = self.noise_floor
        db = 20 * np.log10(rms / self.noise_floor)
        return max(0, db + 60)  # Коррекция для универсальных звуков
    
    def interpolate_distance(self, db_level):
        """Интерполяция расстояния"""
        for i, (table_db, table_dist, desc) in enumerate(self.distance_table):
            if db_level >= table_db:
                if i == 0:
                    return table_dist, desc
                
                prev_db, prev_dist, prev_desc = self.distance_table[i-1]
                ratio = (db_level - table_db) / (prev_db - table_db)
                distance = table_dist + ratio * (prev_dist - table_dist)
                return distance, f"Между {prev_desc.lower()} и {desc.lower()}"
        
        return self.distance_table[-1][1], self.distance_table[-1][2]
    
    def analyze_frequency_content(self, audio_data):
        """Анализ частотного содержания"""
        # FFT
        fft = np.fft.fft(audio_data)
        magnitude = np.abs(fft[:len(fft)//2])
        freqs = np.fft.fftfreq(len(audio_data), 1/self.SAMPLE_RATE)[:len(fft)//2]
        
        # Частотные диапазоны
        low_mask = (freqs >= 20) & (freqs < 300)     # Низкие частоты
        mid_mask = (freqs >= 300) & (freqs < 2000)   # Средние частоты
        high_mask = (freqs >= 2000) & (freqs < 8000) # Высокие частоты
        
        low_energy = np.mean(magnitude[low_mask]) if np.any(low_mask) else 0
        mid_energy = np.mean(magnitude[mid_mask]) if np.any(mid_mask) else 0
        high_energy = np.mean(magnitude[high_mask]) if np.any(high_mask) else 0
        
        total_energy = low_energy + mid_energy + high_energy
        
        if total_energy < 1e-8:
            return "тишина", "🔇", 0.0
        
        # Нормализация
        low_ratio = low_energy / total_energy
        mid_ratio = mid_energy / total_energy
        high_ratio = high_energy / total_energy
        
        # Пиковая частота
        peak_freq_idx = np.argmax(magnitude)
        peak_frequency = freqs[peak_freq_idx] if peak_freq_idx < len(freqs) else 0
        
        # Классификация звука
        if mid_ratio > 0.6 and 300 <= peak_frequency <= 3000:
            return "речь/голос", "🗣️", total_energy
        elif low_ratio > 0.5 and total_energy > self.adaptive_energy_threshold * 3:
            return "удар/басы", "🥁", total_energy
        elif high_ratio > 0.4 and peak_frequency > 2000:
            return "резкий/высокий", "⚡", total_energy
        elif mid_ratio > 0.4 and high_ratio > 0.2:
            return "музыка/мелодия", "🎵", total_energy
        elif low_ratio > 0.6 and peak_frequency < 200:
            return "гул/низкий тон", "🌊", total_energy
        elif total_energy > self.adaptive_energy_threshold * 5:
            return "громкий звук", "📢", total_energy
        else:
            return "смешанный звук", "🔊", total_energy
    
    def detect_sound_pattern(self, audio_data):
        """Детекция паттернов звука"""
        volume = np.abs(audio_data).mean()
        energy = np.sqrt(np.mean(audio_data**2))
        
        # Обновляем историю
        self.volume_history.append(volume)
        
        # Проверка основных критериев
        volume_check = volume > self.adaptive_volume_threshold
        energy_check = energy > self.adaptive_energy_threshold
        
        # Проверка на пик
        if len(self.volume_history) >= 5:
            recent_avg = np.mean(list(self.volume_history)[-5:])
            volume_spike = volume > recent_avg * 2.0
        else:
            volume_spike = volume > self.background_volume * 5
        
        # Проверка интервала
        current_time = time.time()
        time_check = (current_time - self.last_detection_time) > self.min_interval
        
        # Результат
        is_significant = (volume_check and energy_check) or volume_spike
        
        return is_significant and time_check, volume, energy
    
    def format_distance(self, distance):
        """Форматирование расстояния"""
        if distance < 1.0:
            return f"{distance*100:.0f}см"
        elif distance < 10.0:
            return f"{distance:.1f}м"
        else:
            return f"{distance:.0f}м"
    
    def get_distance_icon(self, distance):
        """Иконка расстояния"""
        if distance < 0.5:
            return "🔥"
        elif distance < 1.5:
            return "📍"
        elif distance < 3.0:
            return "📌"
        elif distance < 5.0:
            return "📡"
        else:
            return "🌫️"
    
    def create_visual_indicators(self, volume, energy, distance):
        """Визуальные индикаторы"""
        # Громкость
        vol_bar_len = 25
        vol_filled = int((min(volume, 0.1) / 0.1) * vol_bar_len)
        vol_bar = f"🔊 |{'█' * vol_filled}{'░' * (vol_bar_len - vol_filled)}| {volume:.4f}"
        
        # Энергия
        energy_bar_len = 25
        energy_filled = int((min(energy, 0.1) / 0.1) * energy_bar_len)
        energy_bar = f"⚡ |{'█' * energy_filled}{'░' * (energy_bar_len - energy_filled)}| {energy:.4f}"
        
        # Расстояние
        dist_bar_len = 30
        dist_pos = int((min(distance, 10) / 10) * dist_bar_len)
        dist_bar = f"📏 |{'░' * dist_pos}●{'░' * (dist_bar_len - dist_pos)}| {self.format_distance(distance)}"
        
        return vol_bar, energy_bar, dist_bar
    
    def display_detection(self, volume, energy, db_level, distance, description, 
                         sound_type, sound_icon, frequency_energy):
        """Отображение детекции"""
        self.detections_count += 1
        current_time = datetime.now().strftime("%H:%M:%S")
        distance_icon = self.get_distance_icon(distance)
        
        # Создание индикаторов
        vol_bar, energy_bar, dist_bar = self.create_visual_indicators(volume, energy, distance)
        
        print("\n")
        print(f"🔔 === ЗВУК #{self.detections_count} ОБНАРУЖЕН === {current_time}")
        
        # Индикаторы
        print(vol_bar)
        print(energy_bar)
        print(dist_bar)
        
        # Информация о звуке
        print(f"\n{sound_icon} Тип звука:     {sound_type.upper()}")
        print(f"📊 Громкость:     {volume:.4f} (порог: {self.adaptive_volume_threshold:.4f})")
        print(f"⚡ Энергия:       {energy:.4f} (порог: {self.adaptive_energy_threshold:.4f})")
        print(f"🎵 Частотная энергия: {frequency_energy:.6f}")
        print(f"📢 Децибелы:      {db_level:.1f} дБ")
        print(f"📏 Расстояние:    {distance_icon} {self.format_distance(distance)}")
        print(f"📝 Описание:      {description}")
        print(f"📈 Статистика:    {self.detections_count} детекций за сессию")
        
        # Соотношение к фону
        volume_ratio = volume / self.background_volume
        print(f"📶 Усиление над фоном: {volume_ratio:.1f}x")
        
        print("=" * 60)
    
    def show_activity_dot(self):
        """Показать активность"""
        print(".", end="", flush=True)
    
    def audio_callback(self, indata, frames, time, status):
        """Callback обработки аудио"""
        if status:
            print(f"⚠️ Статус: {status}")
        
        audio_data = indata[:, 0].copy()
        self.frames_processed += 1
        
        try:
            # Детекция звука
            is_sound, volume, energy = self.detect_sound_pattern(audio_data)
            
            if is_sound:
                # Обновляем время последней детекции
                self.last_detection_time = time.time()
                
                # Анализ частот
                sound_type, sound_icon, freq_energy = self.analyze_frequency_content(audio_data)
                
                # Расчет параметров
                db_level = self.calculate_decibels(audio_data)
                distance, description = self.interpolate_distance(db_level)
                
                # Сглаживание
                self.db_buffer.append(db_level)
                self.distance_buffer.append(distance)
                
                if len(self.db_buffer) >= 3:
                    smoothed_db = np.mean(list(self.db_buffer))
                    smoothed_distance = np.median(list(self.distance_buffer))
                else:
                    smoothed_db = db_level
                    smoothed_distance = distance
                
                # Отображение
                self.display_detection(
                    volume, energy, smoothed_db, smoothed_distance, description,
                    sound_type, sound_icon, freq_energy
                )
            else:
                # Показываем активность каждые 50 кадров
                if self.frames_processed % 50 == 0:
                    self.show_activity_dot()
                    
        except Exception as e:
            print(f"⚠️ Ошибка: {e}")
    
    def list_audio_devices(self):
        """Список аудио устройств"""
        print("🎤 Доступные устройства:")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"   {i}: {device['name']} ✅")
    
    def start_detection(self):
        """Запуск детекции"""
        print("\n🎯 === ПАРАМЕТРЫ ПРОСТОГО ДЕТЕКТОРА ===")
        print(f"   📊 Частота: {self.SAMPLE_RATE} Гц")
        print(f"   🔊 Порог громкости: {self.adaptive_volume_threshold:.6f}")
        print(f"   ⚡ Энергетический порог: {self.adaptive_energy_threshold:.6f}")
        print(f"   ⏱️ Минимальный интервал: {self.min_interval}с")
        print(f"   🎯 Цель: ЛЮБЫЕ ЗНАЧИМЫЕ ЗВУКИ")
        print("=" * 50)
        
        self.list_audio_devices()
        
        print(f"\n🎵 Простой детектор готов!")
        print(f"🔊 Издавайте любые звуки - голос, музыку, стуки...")
        print(f"✅ Адаптирован к фоновому шуму: {self.background_volume:.6f}")
        print(f"⏱️ Активность: ", end="", flush=True)
        
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
            print(f"\n\n🛑 Остановка...")
            print(f"📊 Обнаружено звуков: {self.detections_count}")
            print(f"📈 Обработано кадров: {self.frames_processed}")
            if self.frames_processed > 0:
                print(f"🎯 Эффективность: {(self.detections_count/self.frames_processed*100):.2f}%")
            self.running = False
        except Exception as e:
            print(f"❌ Ошибка: {e}")
        finally:
            print("👋 До свидания!")

def main():
    try:
        detector = SimpleUniversalSoundDetector()
        detector.start_detection()
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")

if __name__ == "__main__":
    main()