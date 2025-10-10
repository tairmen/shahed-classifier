import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import sounddevice as sd
import time
from collections import deque
import threading
import sys
import math
from scipy import signal
from scipy.signal import correlate

# Пути
MODEL_PATH = "model/my_sound_model.h5"
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
classifier = tf.keras.models.load_model(MODEL_PATH)
class_names = ["negative", "positive"]

SAMPLE_RATE = 16000
BLOCK_DURATION = 0.5  # Уменьшили для более быстрого отклика
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DURATION)
MIN_CONFIDENCE = 0.8
SMOOTH_WINDOW = 3
RMS_THRESHOLD = 0.008  # Немного снизили порог
CHANNELS = 2

# Константы для расчета направления
SPEED_OF_SOUND = 343.0  # м/с
MIC_DISTANCE = 0.15     # Расстояние между микрофонами (15 см)

# Буферы
prediction_buffer = deque(maxlen=SMOOTH_WINDOW)
volume_buffer_left = deque(maxlen=30)
volume_buffer_right = deque(maxlen=30)
direction_buffer = deque(maxlen=15)
confidence_buffer = deque(maxlen=10)
last_prediction = None
last_time = 0
is_running = True

class AdvancedDirectionDetector:
    def __init__(self, sample_rate=16000, mic_distance=0.15):
        self.sample_rate = sample_rate
        self.mic_distance = mic_distance
        self.max_delay_samples = int((mic_distance / SPEED_OF_SOUND) * sample_rate)
        
    def enhance_signal(self, signal_data):
        """Улучшение сигнала для лучшего определения направления"""
        # Применяем фильтр высоких частот для удаления низкочастотного шума
        sos = signal.butter(4, 100, btype='high', fs=self.sample_rate, output='sos')
        filtered = signal.sosfilt(sos, signal_data)
        
        # Нормализация
        if np.max(np.abs(filtered)) > 0:
            filtered = filtered / np.max(np.abs(filtered))
        
        return filtered
    
    def calculate_direction_advanced(self, left_channel, right_channel):
        """Продвинутый алгоритм определения направления"""
        try:
            # Улучшение сигналов
            left_enhanced = self.enhance_signal(left_channel)
            right_enhanced = self.enhance_signal(right_channel)
            
            # Взаимная корреляция с улучшенными сигналами
            correlation = correlate(left_enhanced, right_enhanced, mode='full')
            
            # Находим максимум корреляции
            lags = signal.correlation_lags(len(left_enhanced), len(right_enhanced), mode='full')
            max_corr_idx = np.argmax(correlation)
            delay_samples = lags[max_corr_idx]
            
            # Ограничиваем задержку физически возможными значениями
            delay_samples = np.clip(delay_samples, -self.max_delay_samples, self.max_delay_samples)
            
            # Конвертируем в время и угол
            delay_time = delay_samples / self.sample_rate
            
            if abs(delay_time) > 0:
                # Вычисляем угол более точно
                sin_theta = (delay_time * SPEED_OF_SOUND) / self.mic_distance
                sin_theta = np.clip(sin_theta, -1, 1)
                
                angle_rad = math.asin(sin_theta)
                angle_deg = math.degrees(angle_rad)
                
                # Определяем качество корреляции
                correlation_quality = correlation[max_corr_idx] / np.max(correlation)
                
                return angle_deg, delay_time, correlation_quality
            else:
                return 0.0, 0.0, 0.0
                
        except Exception as e:
            return 0.0, 0.0, 0.0
    
    def calculate_phase_difference(self, left_channel, right_channel):
        """Вычисляет разность фаз для дополнительного анализа"""
        try:
            # FFT для анализа фаз
            left_fft = np.fft.fft(left_channel)
            right_fft = np.fft.fft(right_channel)
            
            # Находим доминирующую частоту
            freqs = np.fft.fftfreq(len(left_channel), 1/self.sample_rate)
            left_magnitude = np.abs(left_fft)
            
            # Берем частоты от 200 до 2000 Hz (речевой диапазон)
            freq_mask = (freqs > 200) & (freqs < 2000)
            if np.any(freq_mask):
                dominant_freq_idx = np.argmax(left_magnitude[freq_mask])
                actual_idx = np.where(freq_mask)[0][dominant_freq_idx]
                
                # Вычисляем разность фаз
                phase_diff = np.angle(left_fft[actual_idx]) - np.angle(right_fft[actual_idx])
                
                # Конвертируем в угол
                wavelength = SPEED_OF_SOUND / freqs[actual_idx]
                angle_rad = math.asin(np.clip((phase_diff * wavelength) / (2 * math.pi * self.mic_distance), -1, 1))
                angle_deg = math.degrees(angle_rad)
                
                return angle_deg, freqs[actual_idx]
            else:
                return 0.0, 0.0
                
        except Exception:
            return 0.0, 0.0

# Инициализируем детектор
direction_detector = AdvancedDirectionDetector()

def extract_embedding_from_audio(audio):
    scores, embeddings, spectrogram = yamnet_model(audio)
    embedding = tf.reduce_mean(embeddings, axis=0)
    return embedding

def smooth_predictions(new_prediction):
    prediction_buffer.append(new_prediction)
    if len(prediction_buffer) < SMOOTH_WINDOW:
        return None
    
    negative_count = sum(1 for p in prediction_buffer if p == 0)
    positive_count = sum(1 for p in prediction_buffer if p == 1)
    return 0 if negative_count >= positive_count else 1

def get_direction_description_advanced(angle, quality=1.0):
    """Расширенное описание направления с учетом качества"""
    confidence_emoji = "🟢" if quality > 0.7 else "🟡" if quality > 0.4 else "🔴"
    
    if abs(angle) < 5:
        return "📍 ТОЧНО ЦЕНТР", "🎯", confidence_emoji
    elif abs(angle) < 15:
        direction = "↙️ ЛЕВЕЕ" if angle > 0 else "↘️ ПРАВЕЕ"
        return direction, "↙️" if angle > 0 else "↘️", confidence_emoji
    elif abs(angle) < 30:
        direction = "⬅️ ЛЕВО-ЦЕНТР" if angle > 0 else "➡️ ПРАВО-ЦЕНТР"
        return direction, "⬅️" if angle > 0 else "➡️", confidence_emoji
    elif abs(angle) < 50:
        direction = "⬅️ ЛЕВЫЙ" if angle > 0 else "➡️ ПРАВЫЙ"
        return direction, "⬅️" if angle > 0 else "➡️", confidence_emoji
    else:
        direction = "⬅️ КРАЙНЕ ЛЕВЫЙ" if angle > 0 else "➡️ КРАЙНЕ ПРАВЫЙ"
        return direction, "⬅️" if angle > 0 else "➡️", confidence_emoji

def create_3d_direction_visualizer(angle, quality=1.0, width=40):
    """Создает 3D ASCII визуализацию направления"""
    # Основная полоса направления
    center = width // 2
    pos = center + int((angle / 90) * (width // 2))
    pos = np.clip(pos, 0, width - 1)
    
    # Создаем визуализацию с интенсивностью на основе качества
    bar = list("░" * width)
    
    # Размытие на основе качества
    spread = max(1, int(3 * (1 - quality)))
    
    for i in range(max(0, pos - spread), min(width, pos + spread + 1)):
        if i == pos:
            bar[i] = "🔊" if quality > 0.7 else "🔉" if quality > 0.4 else "🔈"
        else:
            bar[i] = "▓" if quality > 0.7 else "▒" if quality > 0.4 else "░"
    
    # Добавляем метки
    bar[0] = "L"
    bar[center] = "|"
    bar[width - 1] = "R"
    
    return "".join(bar)

def create_polar_visualization(angle, distance_estimate=1.0):
    """Создает полярную визуализацию положения источника"""
    # Простая ASCII полярная диаграмма
    rows = 7
    cols = 15
    
    # Центр диаграммы
    center_row = rows // 2
    center_col = cols // 2
    
    # Вычисляем позицию источника
    angle_rad = math.radians(angle)
    source_row = center_row - int(distance_estimate * 2 * math.sin(angle_rad))
    source_col = center_col + int(distance_estimate * 4 * math.cos(angle_rad))
    
    # Ограничиваем в пределах диаграммы
    source_row = np.clip(source_row, 0, rows - 1)
    source_col = np.clip(source_col, 0, cols - 1)
    
    # Создаем диаграмму
    diagram = [["·" for _ in range(cols)] for _ in range(rows)]
    
    # Центр (микрофон)
    diagram[center_row][center_col] = "🎤"
    
    # Источник звука
    diagram[source_row][source_col] = "🔊"
    
    # Рамка
    for i in range(rows):
        diagram[i][0] = diagram[i][cols-1] = "|"
    for j in range(cols):
        diagram[0][j] = diagram[rows-1][j] = "-"
    
    return "\n".join("".join(row) for row in diagram)

def advanced_stats_display():
    """Расширенная статистика с направлением"""
    while is_running:
        if direction_buffer and len(direction_buffer) >= 5:
            recent_data = list(direction_buffer)[-10:]
            avg_angle = np.mean(recent_data)
            angle_std = np.std(recent_data)
            
            # Оценка стабильности
            stability = "🟢 СТАБИЛЬНО" if angle_std < 10 else "🟡 УМЕРЕННО" if angle_std < 20 else "🔴 НЕСТАБИЛЬНО"
            
            # Качество корреляции
            avg_quality = np.mean(confidence_buffer) if confidence_buffer else 0
            
            direction_desc, emoji, quality_emoji = get_direction_description_advanced(avg_angle, avg_quality)
            
            print(f"\n🧭 === РАСШИРЕННЫЙ АНАЛИЗ НАПРАВЛЕНИЯ ===")
            print(f"   Направление: {direction_desc}")
            print(f"   Угол: {avg_angle:.1f}° ± {angle_std:.1f}°")
            print(f"   Стабильность: {stability}")
            print(f"   Качество сигнала: {quality_emoji} {avg_quality:.2f}")
            
            # Визуализация
            direction_viz = create_3d_direction_visualizer(avg_angle, avg_quality)
            print(f"   Позиция: {direction_viz}")
            
            # Полярная диаграмма
            if abs(avg_angle) > 5:  # Показываем только для четких направлений
                polar_viz = create_polar_visualization(avg_angle)
                print(f"   Полярная диаграмма:")
                for line in polar_viz.split('\n'):
                    print(f"     {line}")
            
            print("=" * 60)
            
            for _ in range(3):
                print()
        
        time.sleep(5)

def audio_callback(indata, frames, time_info, status):
    """Продвинутая обработка аудио с направлением"""
    global last_prediction, last_time
    
    if status:
        print(f"\n⚠️ Audio status: {status}")
    
    try:
        # Разделяем каналы
        if indata.shape[1] >= 2:
            left_channel = indata[:, 0].astype(np.float32)
            right_channel = indata[:, 1].astype(np.float32)
        else:
            mono = indata[:, 0].astype(np.float32)
            left_channel = right_channel = mono
        
        # Вычисляем RMS
        left_rms = np.sqrt(np.mean(left_channel**2))
        right_rms = np.sqrt(np.mean(right_channel**2))
        avg_rms = (left_rms + right_rms) / 2
        
        volume_buffer_left.append(left_rms)
        volume_buffer_right.append(right_rms)
        
        # Определяем направление
        if avg_rms > RMS_THRESHOLD:
            angle, delay, quality = direction_detector.calculate_direction_advanced(left_channel, right_channel)
            phase_angle, dominant_freq = direction_detector.calculate_phase_difference(left_channel, right_channel)
            
            # Комбинируем результаты
            combined_angle = (angle + phase_angle) / 2 if abs(phase_angle) < 90 else angle
            
            direction_buffer.append(combined_angle)
            confidence_buffer.append(quality)
        
        if avg_rms < RMS_THRESHOLD:
            return
        
        # Классификация
        mono_audio = (left_channel + right_channel) / 2
        emb = extract_embedding_from_audio(mono_audio)
        pred = classifier.predict(np.expand_dims(emb, axis=0), verbose=0)[0]
        label_idx = np.argmax(pred)
        confidence = pred[label_idx]
        
        smoothed_prediction = smooth_predictions(label_idx)
        
        current_time = time.time()
        if (smoothed_prediction is not None and 
            confidence > MIN_CONFIDENCE and 
            smoothed_prediction != last_prediction and
            current_time - last_time > 1.0):
            
            label = class_names[smoothed_prediction]
            
            # Информация о направлении
            if direction_buffer and confidence_buffer:
                current_angle = direction_buffer[-1]
                signal_quality = confidence_buffer[-1]
                direction_desc, emoji, quality_emoji = get_direction_description_advanced(current_angle, signal_quality)
            else:
                current_angle = 0
                signal_quality = 0
                direction_desc, emoji, quality_emoji = "📍 НЕИЗВЕСТНО", "❓", "🔴"
            
            # Вывод результата
            sys.stdout.write("\r" + " " * 120 + "\r")
            
            result_emoji = "🔴" if label == "negative" else "🟢"
            print(f"\n{result_emoji} === ПРОДВИНУТОЕ ОБНАРУЖЕНИЕ === {result_emoji}")
            print(f"🎧 Класс:         {label.upper()}")
            print(f"📊 Уверенность:   {confidence:.2f} ({confidence*100:.1f}%)")
            print(f"🔊 Громкость:     L:{left_rms:.3f} R:{right_rms:.3f}")
            print(f"🧭 Направление:   {direction_desc}")
            print(f"📐 Угол:          {current_angle:.1f}°")
            print(f"🎯 Качество:      {quality_emoji} {signal_quality:.2f}")
            print(f"⏰ Время:         {time.strftime('%H:%M:%S')}")
            
            # Визуализация направления
            direction_viz = create_3d_direction_visualizer(current_angle, signal_quality)
            print(f"📍 Позиция:       {direction_viz}")
            
            print("=" * 60)
            
            last_prediction = smoothed_prediction
            last_time = current_time
            
    except Exception as e:
        print(f"\n❌ Error in audio callback: {e}")

def main():
    print("🎙️ === ПРОДВИНУТЫЙ ДЕТЕКТОР НАПРАВЛЕНИЯ ЗВУКА ===")
    print("🧭 Улучшенные алгоритмы определения направления")
    print("🔬 Анализ корреляции и разности фаз")
    print("=" * 70)
    
    print(f"\n🔧 Параметры:")
    print(f"   Расстояние между микрофонами: {MIC_DISTANCE*100} см")
    print(f"   Максимальная задержка: {(MIC_DISTANCE/SPEED_OF_SOUND)*1000:.2f} мс")
    print(f"   Разрешение угла: ~{math.degrees(math.asin(1/(SAMPLE_RATE*MIC_DISTANCE/SPEED_OF_SOUND))):.1f}°")
    print("=" * 70)
    print("🎯 Готов к продвинутому анализу направления!")
    print("(Ctrl+C для выхода)\n")
    
    # Запускаем потоки
    stats_thread = threading.Thread(target=advanced_stats_display, daemon=True)
    stats_thread.start()
    
    # Основной аудио поток
    try:
        with sd.InputStream(channels=CHANNELS, samplerate=SAMPLE_RATE, 
                          callback=audio_callback, blocksize=BLOCK_SIZE):
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                global is_running
                is_running = False
                print("\n🛑 Остановка продвинутого детектора...")
                print("👋 До свидания!")
    except Exception as e:
        print(f"❌ Ошибка аудио: {e}")

if __name__ == "__main__":
    main()