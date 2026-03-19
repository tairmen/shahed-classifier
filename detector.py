#!/usr/bin/env python3
"""
Универсальный детектор звуков — Android / Termux версия
Работает БЕЗ полного TensorFlow, только tflite-runtime.

== УСТАНОВКА (Termux) ==========================================
1. Обновляем пакеты:
      pkg update && pkg upgrade -y

2. Устанавливаем системные зависимости:
      pkg install python portaudio -y

3. Устанавливаем Python-зависимости:
      pip install numpy scipy sounddevice tflite-runtime

4. Даём Termux доступ к микрофону:
      Настройки Android → Приложения → Termux → Разрешения → Микрофон
   Или в самом Termux:
      termux-microphone-record -h  (запустит запрос разрешения)

== ПОДГОТОВКА МОДЕЛЕЙ (один раз, на ПК) =======================
5. На Windows/Linux/Mac запустите:
      python convert_yamnet_to_tflite.py
   Получится файл model/yamnet_embeddings.tflite (~3.7 MB)

6. Скопируйте на телефон (через USB / adb / sftp):
      model/yamnet_embeddings.tflite
      model/my_sound_model.tflite
      distance_calibration.json
      universal_sound_detector_android.py

== ЗАПУСК ======================================================
7. В Termux:
      python universal_sound_detector_android.py

================================================================
"""

import numpy as np
import json
import time
import os
from datetime import datetime
from collections import deque

import scipy.signal

try:
    import sounddevice as sd
except ImportError:
    raise ImportError(
        "sounddevice не установлен.\n"
        "Выполните: pkg install portaudio && pip install sounddevice"
    )

try:
    import onnxruntime as ort
    _RUNTIME = "onnxruntime"
except ImportError:
    raise ImportError(
        "onnxruntime не установлен.\n"
        "Выполните: pkg install tur-repo && pkg install onnxruntime"
    )


class UniversalSoundDetectorAndroid:
    def __init__(self, calibration_file=None, num_threads=4, input_device=None):
        print("🎵 === УНИВЕРСАЛЬНЫЙ ДЕТЕКТОР ЗВУКОВ (Android/TFLite) ===")
        print(f"✅ Используется: {_RUNTIME}")
        print("= " * 25)

        self._base_dir = os.path.dirname(os.path.abspath(__file__))
        self.input_device = input_device  # None = default

        if calibration_file is None:
            calibration_file = os.path.join(self._base_dir, 'distance_calibration.json')

        self.SAMPLE_RATE = 16000
        self.CAPTURE_RATE = 48000   # нативная частота Android OpenSL ES
        self.CAPTURE_CHANNELS = 2   # OpenSL ES требует стерео на большинстве Android
        self.WINDOW_SIZE = 0.975
        self.CHUNK_SIZE = int(self.SAMPLE_RATE * self.WINDOW_SIZE)   # 15600
        self.CAPTURE_CHUNK = int(self.CAPTURE_RATE * self.WINDOW_SIZE)  # ~46800
        self.CONFIDENCE_THRESHOLD = 0.6
        self.MIN_VOLUME = 0.002
        self.num_threads = num_threads

        self.prediction_buffer = deque(maxlen=3)
        self.volume_buffer = deque(maxlen=10)
        self.distance_buffer = deque(maxlen=5)
        self.db_buffer = deque(maxlen=3)

        self.positive_detections = 0
        self.total_attempts = 0
        self.debug_counter = 0
        self.running = False
        self.last_positive_detection = None
        self.environment_type = "outdoor"

        self.load_calibration_data(calibration_file)
        self.load_models()
        self.calibrate_microphone()

    # ------------------------------------------------------------------ #
    #  КАЛИБРОВКА                                                          #
    # ------------------------------------------------------------------ #

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
            self._create_default_calibration()

    def _create_default_calibration(self):
        self.distance_table = [
            (70, 0.3, "Очень близко"),
            (65, 0.5, "Близко"),
            (60, 1.0, "Рядом"),
            (55, 1.5, "Близко — тихий звук"),
            (50, 2.0, "На расстоянии"),
            (45, 3.0, "Далеко — приглушённый"),
            (40, 4.0, "Далеко — слабый"),
            (35, 5.0, "Очень далеко"),
            (30, 7.0, "Крайне далеко"),
            (25, 10.0, "На пределе слышимости"),
        ]

    # ------------------------------------------------------------------ #
    #  ЗАГРУЗКА МОДЕЛЕЙ                                                   #
    # ------------------------------------------------------------------ #

    def load_models(self):
        yamnet_path = os.path.join(self._base_dir, 'model', 'yamnet_embeddings.onnx')
        classifier_path = os.path.join(self._base_dir, 'model', 'my_sound_model.onnx')

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = self.num_threads

        # YAMNet — извлечение эмбеддингов
        print(f"🔄 Загрузка YAMNet ONNX: {yamnet_path}")
        if not os.path.exists(yamnet_path):
            raise FileNotFoundError(
                f"Файл не найден: {yamnet_path}\n"
                "Запустите на ПК: python -m tf2onnx.convert --tflite model/yamnet_embeddings.tflite --output model/yamnet_embeddings.onnx"
            )
        self.yamnet_session = ort.InferenceSession(yamnet_path, sess_options)
        self.yamnet_input_name = self.yamnet_session.get_inputs()[0].name
        self.yamnet_output_name = self.yamnet_session.get_outputs()[0].name
        print(f"   Вход:  {self.yamnet_session.get_inputs()[0].name}  {self.yamnet_session.get_inputs()[0].shape}")
        print(f"   Выход: {self.yamnet_session.get_outputs()[0].name}  {self.yamnet_session.get_outputs()[0].shape}")
        print("✅ YAMNet загружен")

        # Классификатор positive/negative
        print(f"🔄 Загрузка классификатора: {classifier_path}")
        if not os.path.exists(classifier_path):
            raise FileNotFoundError(
                f"Файл не найден: {classifier_path}\n"
                "Запустите на ПК: python -m tf2onnx.convert --tflite model/my_sound_model.tflite --output model/my_sound_model.onnx"
            )
        self.classifier_session = ort.InferenceSession(classifier_path, sess_options)
        self.classifier_input_name = self.classifier_session.get_inputs()[0].name
        self.classifier_output_name = self.classifier_session.get_outputs()[0].name
        print("✅ Классификатор загружен")

    # ------------------------------------------------------------------ #
    #  МИКРОФОН                                                            #
    # ------------------------------------------------------------------ #

    def calibrate_microphone(self):
        print("🎙️ Калибровка микрофона (3 секунды тишины)...")
        print("   Если ошибка микрофона — дайте разрешение:")
        print("   Настройки Android → Приложения → Termux → Разрешения → Микрофон")
        silence_samples = []
        for i in range(3, 0, -1):
            print(f"   {i}...")
            time.sleep(1)
            try:
                audio = sd.rec(
                    int(self.CAPTURE_RATE * 0.5),
                    samplerate=self.CAPTURE_RATE,
                    channels=self.CAPTURE_CHANNELS,
                    dtype=np.float32,
                    device=self.input_device
                )
                sd.wait()
                mono = audio.mean(axis=1) if audio.ndim > 1 else audio.flatten()
                silence_samples.append(np.abs(mono).mean())
            except Exception as e:
                print(f"   ⚠️ Ошибка записи: {e}")
                silence_samples.append(1e-4)
        self.noise_floor = max(float(np.mean(silence_samples)), 1e-8)
        print(f"✅ Уровень шума: {self.noise_floor:.6f}")

    # ------------------------------------------------------------------ #
    #  АУДИО-ОБРАБОТКА                                                     #
    # ------------------------------------------------------------------ #

    def calculate_decibels(self, audio_data):
        rms = float(np.sqrt(np.mean(audio_data ** 2)))
        rms = max(rms, self.noise_floor)
        db = 20.0 * np.log10(rms / self.noise_floor)
        return float(max(0.0, db + 60.0))

    def classify_audio(self, audio_data):
        try:
            # 1. Привести к нужному размеру
            if len(audio_data) != self.CHUNK_SIZE:
                audio_data = scipy.signal.resample(audio_data, self.CHUNK_SIZE)
            audio_data = audio_data.astype(np.float32)

            # 2. Подготовить вход для YAMNet (форма зависит от конвертации)
            yamnet_in_shape = self.yamnet_session.get_inputs()[0].shape
            if len(yamnet_in_shape) == 2:          # [1, 15600]
                yamnet_input = audio_data.reshape(1, -1)
            else:                                   # [15600]
                yamnet_input = audio_data

            # 3. Получить эмбеддинг
            embedding = self.yamnet_session.run(
                [self.yamnet_output_name],
                {self.yamnet_input_name: yamnet_input}
            )[0]

            # 4. Нормализовать форму → [1, 1024]
            if embedding.ndim == 1:
                embedding = embedding.reshape(1, -1)
            elif embedding.ndim == 2 and embedding.shape[0] > 1:
                embedding = np.mean(embedding, axis=0, keepdims=True)
            embedding = embedding.astype(np.float32)

            # 5. Классифицировать
            output = self.classifier_session.run(
                [self.classifier_output_name],
                {self.classifier_input_name: embedding}
            )[0]

            confidence = float(np.max(output))
            predicted_class = "positive" if output[0][1] > output[0][0] else "negative"
            return predicted_class, confidence

        except Exception as e:
            print(f"\n⚠️ Ошибка классификации: {e}")
            return "error", 0.0

    def interpolate_distance(self, db_level):
        for i, (table_db, table_dist, desc) in enumerate(self.distance_table):
            if db_level >= table_db:
                if i == 0:
                    return table_dist, desc
                prev_db, prev_dist, prev_desc = self.distance_table[i - 1]
                ratio = (db_level - table_db) / (prev_db - table_db)
                distance = table_dist + ratio * (prev_dist - table_dist)
                return distance, f"Между «{prev_desc.lower()}» и «{desc.lower()}»"
        return self.distance_table[-1][1], self.distance_table[-1][2]

    def smooth_values(self, db_level, distance):
        self.db_buffer.append(db_level)
        self.distance_buffer.append(distance)
        if len(self.db_buffer) >= 3:
            return float(np.mean(self.db_buffer)), float(np.median(self.distance_buffer))
        return db_level, distance

    # ------------------------------------------------------------------ #
    #  ОТОБРАЖЕНИЕ                                                         #
    # ------------------------------------------------------------------ #

    def format_distance(self, distance):
        if distance < 1.0:
            return f"{distance * 100:.0f}см"
        elif distance < 10.0:
            return f"{distance:.1f}м"
        else:
            return f"{distance:.0f}м"

    def get_distance_icon(self, distance):
        if distance < 0.5:   return "🔥"
        elif distance < 1.5: return "📍"
        elif distance < 3.0: return "📌"
        elif distance < 5.0: return "📡"
        else:                return "🌫️"

    def create_progress_bars(self, volume, distance, confidence):
        W = 20
        vol_filled = int((min(volume, 0.1) / 0.1) * W)
        vol_bar = f"🔊 |{'█' * vol_filled}{'░' * (W - vol_filled)}| {volume:.3f}"

        D = 25
        dist_pos = int((min(distance, 10) / 10) * D)
        dist_bar = f"📏 |{'░' * dist_pos}●{'░' * (D - dist_pos)}| {self.format_distance(distance)}"

        conf_filled = int(confidence * W)
        conf_bar = f"🎯 |{'█' * conf_filled}{'░' * (W - conf_filled)}| {confidence:.2f}"

        return vol_bar, dist_bar, conf_bar

    def display_positive_detection(self, volume, db_level, distance, description, confidence):
        current_time = datetime.now().strftime("%H:%M:%S")
        icon = self.get_distance_icon(distance)
        vol_bar, dist_bar, conf_bar = self.create_progress_bars(volume, distance, confidence)

        print("\n")
        print(f"🟢 === ЗВУК ОБНАРУЖЕН #{self.positive_detections} === {current_time}")
        print(vol_bar)
        print(dist_bar)
        print(conf_bar)
        print(f"\n🎯 Статус:       ✅ POSITIVE")
        print(f"📊 Уверенность:  {confidence:.2f} ({confidence * 100:.1f}%)")
        print(f"📢 Децибелы:     {db_level:.1f} дБ")
        print(f"📏 Расстояние:   {icon} {self.format_distance(distance)}")
        print(f"📝 Описание:     {description}")
        print(f"📈 Статистика:   {self.positive_detections} из {self.total_attempts}")
        print("=" * 60)

        self.last_positive_detection = {
            'confidence': confidence,
            'distance': distance,
            'db_level': db_level,
            'time': current_time,
        }

    # ------------------------------------------------------------------ #
    #  CALLBACK                                                            #
    # ------------------------------------------------------------------ #

    def audio_callback(self, indata, frames, t, status):
        if status:
            print(f"\n⚠️ Статус аудио: {status}")

        # Стерео → моно (среднее двух каналов)
        audio_data = indata.mean(axis=1).copy() if indata.ndim > 1 else indata[:, 0].copy()
        volume = float(np.abs(audio_data).mean())
        self.total_attempts += 1

        if volume < self.MIN_VOLUME:
            if self.total_attempts % 100 == 0:
                print(".", end="", flush=True)
            return

        try:
            predicted_class, confidence = self.classify_audio(audio_data)

            # Первые 5 обнаружений любого уровня — для отладки
            if self.debug_counter < 5:
                self.debug_counter += 1
                db = self.calculate_decibels(audio_data)
                print(f"\n🔧 [{self.debug_counter}/5] vol={volume:.4f} db={db:.1f} "
                      f"class={predicted_class} conf={confidence:.3f}")

            if predicted_class == "positive" and confidence >= self.CONFIDENCE_THRESHOLD:
                db_level = self.calculate_decibels(audio_data)
                distance, description = self.interpolate_distance(db_level)
                smoothed_db, smoothed_distance = self.smooth_values(db_level, distance)
                self.positive_detections += 1
                self.display_positive_detection(
                    volume, smoothed_db, smoothed_distance, description, confidence
                )
            else:
                if self.total_attempts % 50 == 0:
                    print(".", end="", flush=True)

        except Exception as e:
            print(f"\n⚠️ Ошибка обработки: {e}")

    # ------------------------------------------------------------------ #
    #  ЗАПУСК                                                              #
    # ------------------------------------------------------------------ #

    def list_audio_devices(self):
        print("🎤 Доступные аудио-устройства:")
        found = False
        for i, dev in enumerate(sd.query_devices()):
            if dev['max_input_channels'] > 0:
                marker = " ← ИСПОЛЬЗУЕТСЯ" if i == self.input_device else ""
                print(f"   {i}: {dev['name']} ✅{marker}")
                found = True
        if not found:
            print("   ❌ Нет доступных устройств! Дайте разрешение на микрофон.")
        if self.input_device is None:
            print(f"   (по умолчанию: {sd.default.device[0]})")
            print("   Чтобы выбрать устройство: detector = UniversalSoundDetectorAndroid(input_device=0)")

    def start_detection(self):
        print("\n🎯 === ПАРАМЕТРЫ ===")
        print(f"   Частота дискретизации: {self.SAMPLE_RATE} Гц")
        print(f"   Порог уверенности:     {self.CONFIDENCE_THRESHOLD}")
        print(f"   Минимальная громкость: {self.MIN_VOLUME}")
        print(f"   Потоков TFLite:        {self.num_threads}")
        print(f"   Рантайм:               {_RUNTIME}")
        print("= " * 25)
        self.list_audio_devices()

        print("\n🎵 Детектор готов! Ожидаю целевые звуки...")
        print("⏱️ Активность: ", end="", flush=True)

        self.running = True
        try:
            with sd.InputStream(
                samplerate=self.CAPTURE_RATE,
                channels=self.CAPTURE_CHANNELS,
                callback=self.audio_callback,
                blocksize=self.CAPTURE_CHUNK,
                dtype=np.float32,
                device=self.input_device,
            ):
                while self.running:
                    time.sleep(0.1)

        except KeyboardInterrupt:
            print(f"\n\n🛑 Остановка...")
            print(f"📊 Обнаружено целевых звуков: {self.positive_detections}")
            total = max(self.total_attempts, 1)
            print(f"📈 Эффективность: {self.positive_detections}/{self.total_attempts} "
                  f"= {self.positive_detections / total * 100:.1f}%")
            self.running = False

        except Exception as e:
            print(f"\n❌ Ошибка: {e}")

        finally:
            print("👋 До свидания!")


def main():
    try:
        detector = UniversalSoundDetectorAndroid(num_threads=4)
        detector.start_detection()
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")


if __name__ == "__main__":
    main()
