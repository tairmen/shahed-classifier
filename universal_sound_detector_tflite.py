#!/usr/bin/env python3
"""
🎯 Универсальный детектор звуков на TFLite
Работает с моделями my_sound_model.tflite и yamnet_local (SavedModel)
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

class UniversalSoundDetectorTFLite:
    def __init__(self, calibration_file='distance_calibration.json'):
        print("🎵 === УНИВЕРСАЛЬНЫЙ ДЕТЕКТОР ЗВУКОВ (TFLite)===" )
        print("🎯 Определение расстояния до любого источника звука")
        print("✅ Показ информации только при POSITIVE детекции")
        print("= " * 25)
        
        self.SAMPLE_RATE = 16000
        self.WINDOW_SIZE = 0.975
        self.CHUNK_SIZE = int(self.SAMPLE_RATE * self.WINDOW_SIZE)
        self.CONFIDENCE_THRESHOLD = 0.6
        self.MIN_VOLUME = 0.002
        self.SMOOTHING_FRAMES = 3
        self.prediction_buffer = deque(maxlen=self.SMOOTHING_FRAMES)
        self.volume_buffer = deque(maxlen=10)
        self.distance_buffer = deque(maxlen=5)
        self.db_buffer = deque(maxlen=3)
        self.positive_detections = 0
        self.total_attempts = 0
        self.debug_counter = 0
        self.load_calibration_data(calibration_file)
        self.load_models()
        self.calibrate_microphone()
        self.running = False
        self.last_positive_detection = None
        self.environment_type = "indoor"

    def load_calibration_data(self, filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                self.calibration = json.load(f)
            self.distance_table = []
            mapping = self.calibration['calibration_data']['distance_db_mapping']
            for distance_str, data in mapping.items():
                distance = float(distance_str)
                db = data['db']
                self.distance_table.append((db, distance, data['description']))
            self.distance_table.sort(key=lambda x: x[0], reverse=True)
        except Exception:
            self.create_default_calibration()

    def create_default_calibration(self):
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

    def load_models(self):
        print("🔄 Загрузка YAMNet (локально)...")
        yamnet_path = './yamnet_local'
        if os.path.exists(yamnet_path):
            self.yamnet_model = tf.saved_model.load(yamnet_path)
            print("✅ YAMNet загружен локально из ./yamnet_local")
        else:
            print("⚠️ Локальная модель не найдена, загружаем из интернета...")
            self.yamnet_model = hub.load('https://www.kaggle.com/models/google/yamnet/frameworks/TensorFlow2/variations/yamnet/versions/1')
            print("✅ YAMNet загружен из интернета")
        print("🔄 Загрузка TFLite классификатора...")
        tflite_path = 'model/my_sound_model.tflite'
        if not os.path.exists(tflite_path):
            raise FileNotFoundError(f"TFLite модель не найдена: {tflite_path}")
        self.tflite_interpreter = tf.lite.Interpreter(model_path=tflite_path)
        self.tflite_interpreter.allocate_tensors()
        self.input_details = self.tflite_interpreter.get_input_details()
        self.output_details = self.tflite_interpreter.get_output_details()
        print("✅ TFLite модель загружена")

    def calibrate_microphone(self):
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
        print(f"✅ Уровень шума: {self.noise_floor:.6f}")

    def calculate_decibels(self, audio_data):
        rms = np.sqrt(np.mean(audio_data**2))
        if rms < self.noise_floor:
            rms = self.noise_floor
        db = 20 * np.log10(rms / self.noise_floor)
        db = max(0, db + 60)
        return db

    def interpolate_distance(self, db_level):
        for i, (table_db, table_dist, desc) in enumerate(self.distance_table):
            if db_level >= table_db:
                if i == 0:
                    return table_dist, desc
                prev_db, prev_dist, prev_desc = self.distance_table[i-1]
                ratio = (db_level - table_db) / (prev_db - table_db)
                distance = table_dist + ratio * (prev_dist - table_dist)
                return distance, f"Между {prev_desc.lower()} и {desc.lower()}"
        return self.distance_table[-1][1], self.distance_table[-1][2]

    def classify_audio(self, audio_data):
        if self.tflite_interpreter is None or self.yamnet_model is None:
            return "unknown", 0.5
        try:
            if len(audio_data) != self.CHUNK_SIZE:
                resampled_audio = scipy.signal.resample(audio_data, self.CHUNK_SIZE)
            else:
                resampled_audio = audio_data
            resampled_audio = np.float32(resampled_audio)
            scores, embeddings, spectrogram = self.yamnet_model(resampled_audio)
            mean_embedding = np.mean(embeddings.numpy(), axis=0)
            mean_embedding = np.expand_dims(mean_embedding, axis=0).astype(np.float32)
            self.tflite_interpreter.set_tensor(self.input_details[0]['index'], mean_embedding)
            self.tflite_interpreter.invoke()
            output = self.tflite_interpreter.get_tensor(self.output_details[0]['index'])
            confidence = float(np.max(output))
            predicted_class = "positive" if output[0][1] > output[0][0] else "negative"
            return predicted_class, confidence
        except Exception as e:
            print(f"⚠️ Ошибка классификации: {e}")
            return "error", 0.0

    def smooth_values(self, db_level, distance):
        self.db_buffer.append(db_level)
        self.distance_buffer.append(distance)
        if len(self.db_buffer) >= 3:
            smoothed_db = np.mean(list(self.db_buffer))
            smoothed_distance = np.median(list(self.distance_buffer))
            return smoothed_db, smoothed_distance
        return db_level, distance

    def format_distance(self, distance):
        if distance < 1.0:
            return f"{distance*100:.0f}см"
        elif distance < 10.0:
            return f"{distance:.1f}м"
        else:
            return f"{distance:.0f}м"

    def get_distance_icon(self, distance):
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

    def create_progress_bars(self, volume, distance, confidence):
        vol_bar_len = 25
        vol_filled = int((min(volume, 0.1) / 0.1) * vol_bar_len)
        vol_bar = f"🔊 |{'█' * vol_filled}{'░' * (vol_bar_len - vol_filled)}| {volume:.3f}"
        dist_bar_len = 30
        dist_pos = int((min(distance, 10) / 10) * dist_bar_len)
        dist_bar = f"📏 |{'░' * dist_pos}●{'░' * (dist_bar_len - dist_pos)}| {self.format_distance(distance)}"
        conf_bar_len = 20
        conf_filled = int(confidence * conf_bar_len)
        conf_bar = f"🎯 |{'█' * conf_filled}{'░' * (conf_bar_len - conf_filled)}| {confidence:.2f}"
        return vol_bar, dist_bar, conf_bar

    def show_status_dot(self):
        print(".", end="", flush=True)

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f"⚠️ Статус аудио: {status}")
        audio_data = indata[:, 0].copy()
        volume = np.abs(audio_data).mean()
        self.total_attempts += 1
        if volume < self.MIN_VOLUME:
            if self.total_attempts % 100 == 0:
                self.show_status_dot()
            return
        try:
            predicted_class, confidence = self.classify_audio(audio_data)
            if self.debug_counter < 5 and volume > self.MIN_VOLUME:
                self.debug_counter += 1
                db_level = self.calculate_decibels(audio_data)
                print(f"\n🔧 ОТЛАДКА #{self.debug_counter}: volume={volume:.4f}, db={db_level:.1f}, class={predicted_class}, conf={confidence:.3f}")
            if predicted_class == "positive" and confidence >= self.CONFIDENCE_THRESHOLD:
                db_level = self.calculate_decibels(audio_data)
                distance, description = self.interpolate_distance(db_level)
                smoothed_db, smoothed_distance = self.smooth_values(db_level, distance)
                self.positive_detections += 1
                self.display_positive_detection(
                    volume, smoothed_db, smoothed_distance, description,
                    confidence
                )
            else:
                if self.total_attempts % 50 == 0:
                    self.show_status_dot()
        except Exception as e:
            print(f"⚠️ Ошибка обработки: {e}")

    def display_positive_detection(self, volume, db_level, distance, description, confidence):
        current_time = datetime.now().strftime("%H:%M:%S")
        distance_icon = self.get_distance_icon(distance)
        vol_bar, dist_bar, conf_bar = self.create_progress_bars(volume, distance, confidence)
        print("\n")
        print(f"🟢 === ЗВУК ОБНАРУЖЕН #{self.positive_detections} === {current_time}")
        print(vol_bar)
        print(dist_bar)
        print(conf_bar)
        print(f"\n🎯 Статус:        ✅ POSITIVE (целевой звук)")
        print(f"📊 Уверенность:   {confidence:.2f} ({confidence*100:.1f}%)")
        print(f"📢 Децибелы:      {db_level:.1f} дБ")
        print(f"📏 Расстояние:    {distance_icon} {self.format_distance(distance)}")
        print(f"📝 Описание:      {description}")
        print(f"🏠 Окружение:     {self.environment_type.upper()}")
        print(f"📈 Статистика:    {self.positive_detections} детекций из {self.total_attempts} попыток")
        print("=" * 70)
        self.last_positive_detection = {
            'confidence': confidence,
            'distance': distance,
            'db_level': db_level,
            'time': current_time
        }

    def list_audio_devices(self):
        print("🎤 Доступные аудио устройства:")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"   {i}: {device['name']} ✅")

    def start_detection(self):
        print("\n🎯 === ПАРАМЕТРЫ УНИВЕРСАЛЬНОГО ДЕТЕКТОРА (TFLite) ===")
        print(f"   📊 Частота дискретизации: {self.SAMPLE_RATE} Гц")
        print(f"   🎚️ Минимальная уверенность: {self.CONFIDENCE_THRESHOLD}")
        print(f"   🎚️ Минимальная громкость: {self.MIN_VOLUME}")
        print(f"   🏠 Окружение: {self.environment_type}")
        print(f"   🤖 AI классификация: ✅ (TFLite)")
        print(f"   🧠 YAMNet (feature extractor): {'✅' if self.yamnet_model else '❌'}")
        print(f"   🎯 Цель обнаружения: ВАШИ СПЕЦИФИЧЕСКИЕ ЗВУКИ")
        print(f"   📊 Модель: TFLite")
        print("= " * 25)
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
        detector = UniversalSoundDetectorTFLite()
        detector.start_detection()
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")

if __name__ == "__main__":
    main()
