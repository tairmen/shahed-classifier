import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import sounddevice as sd
import time
from collections import deque
import threading
import sys
import math

# Пути
MODEL_PATH = "model/my_sound_model.h5"
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
classifier = tf.keras.models.load_model(MODEL_PATH)
class_names = ["negative", "positive"]

SAMPLE_RATE = 16000
BLOCK_DURATION = 1.0
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DURATION)
MIN_CONFIDENCE = 0.8
SMOOTH_WINDOW = 3
RMS_THRESHOLD = 0.01
CHANNELS = 2  # Стерео для определения направления

# Константы для расчета направления
SPEED_OF_SOUND = 343.0  # м/с при 20°C
MIC_DISTANCE = 0.1      # Расстояние между микрофонами в метрах (10 см)

# Буферы для данных
prediction_buffer = deque(maxlen=SMOOTH_WINDOW)
volume_buffer_left = deque(maxlen=20)
volume_buffer_right = deque(maxlen=20)
direction_buffer = deque(maxlen=10)
last_prediction = None
last_time = 0
is_running = True

def extract_embedding_from_audio(audio):
    """Извлекает эмбеддинг из моно аудио для YAMNet"""
    scores, embeddings, spectrogram = yamnet_model(audio)
    embedding = tf.reduce_mean(embeddings, axis=0)
    return embedding

def smooth_predictions(new_prediction):
    """Сглаживает предсказания"""
    prediction_buffer.append(new_prediction)
    if len(prediction_buffer) < SMOOTH_WINDOW:
        return None
    
    negative_count = sum(1 for p in prediction_buffer if p == 0)
    positive_count = sum(1 for p in prediction_buffer if p == 1)
    return 0 if negative_count >= positive_count else 1

def calculate_direction(left_channel, right_channel):
    """
    Вычисляет направление звука на основе разности времени прихода (TDOA)
    """
    try:
        # Находим взаимную корреляцию между каналами
        correlation = np.correlate(left_channel, right_channel, mode='full')
        
        # Находим пик корреляции
        max_corr_idx = np.argmax(correlation)
        
        # Вычисляем задержку в семплах
        delay_samples = max_corr_idx - len(right_channel) + 1
        
        # Преобразуем в задержку времени
        delay_time = delay_samples / SAMPLE_RATE
        
        # Вычисляем угол (приблизительно для простой геометрии)
        if abs(delay_time) > 0:
            # Максимальная задержка для нашего расстояния между микрофонами
            max_delay = MIC_DISTANCE / SPEED_OF_SOUND
            
            # Нормализуем задержку
            normalized_delay = np.clip(delay_time / max_delay, -1, 1)
            
            # Вычисляем угол в градусах (-90° до +90°)
            angle_rad = math.asin(normalized_delay)
            angle_deg = math.degrees(angle_rad)
            
            return angle_deg, delay_time
        else:
            return 0.0, 0.0
            
    except Exception as e:
        return 0.0, 0.0

def get_direction_description(angle):
    """Преобразует угол в понятное описание направления"""
    if abs(angle) < 10:
        return "📍 ЦЕНТР", "🎯"
    elif angle > 60:
        return "👈 ЛЕВЫЙ", "⬅️"
    elif angle > 30:
        return "↖️ ЛЕВО-ЦЕНТР", "↖️"
    elif angle > 10:
        return "↙️ ЛЕВЕЕ", "↙️"
    elif angle < -60:
        return "👉 ПРАВЫЙ", "➡️"
    elif angle < -30:
        return "↗️ ПРАВО-ЦЕНТР", "↗️"
    elif angle < -10:
        return "↘️ ПРАВЕЕ", "↘️"
    else:
        return "📍 ЦЕНТР", "🎯"

def create_direction_visualizer(angle, max_width=30):
    """Создает ASCII визуализацию направления звука"""
    # Нормализуем угол от -90 до +90 в позицию от 0 до max_width
    normalized_pos = ((angle + 90) / 180) * max_width
    pos = int(np.clip(normalized_pos, 0, max_width - 1))
    
    # Создаем визуализацию
    bar = list("░" * max_width)
    bar[pos] = "🔊"
    
    # Добавляем метки
    left_mark = "L"
    center_mark = "C"
    right_mark = "R"
    
    bar[0] = left_mark
    bar[max_width // 2] = center_mark
    bar[max_width - 1] = right_mark
    
    return "".join(bar)

def create_stereo_volume_meter(left_vol, right_vol, width=20):
    """Создает стерео индикатор громкости"""
    # Нормализация для отображения
    left_norm = min(1.0, np.log10(left_vol * 1000 + 1) / 3) if left_vol > 0 else 0
    right_norm = min(1.0, np.log10(right_vol * 1000 + 1) / 3) if right_vol > 0 else 0
    
    left_filled = int(left_norm * width)
    right_filled = int(right_norm * width)
    
    left_bar = "█" * left_filled + "░" * (width - left_filled)
    right_bar = "█" * right_filled + "░" * (width - right_filled)
    
    # Балансировка
    balance = left_vol - right_vol
    if abs(balance) < 0.005:
        balance_indicator = "⚖️ БАЛАНС"
    elif balance > 0:
        balance_indicator = "⬅️ ЛЕВЕЕ"
    else:
        balance_indicator = "➡️ ПРАВЕЕ"
    
    return f"L |{left_bar}| {left_vol:.3f}\nR |{right_bar}| {right_vol:.3f}\n{balance_indicator}"

def display_direction_stats():
    """Отображает статистику направления"""
    while is_running:
        if direction_buffer and len(direction_buffer) >= 3:
            recent_angles = list(direction_buffer)[-5:]
            avg_angle = np.mean(recent_angles)
            
            direction_desc, emoji = get_direction_description(avg_angle)
            direction_viz = create_direction_visualizer(avg_angle)
            
            print(f"\n🧭 === АНАЛИЗ НАПРАВЛЕНИЯ ===")
            print(f"   Направление: {direction_desc}")
            print(f"   Угол: {avg_angle:.1f}°")
            print(f"   Визуализация: {direction_viz}")
            print(f"   {emoji} Источник звука")
            print("=" * 50)
            
            for _ in range(5):
                print()
        
        time.sleep(4)

def check_audio_devices():
    """Проверяет доступные аудио устройства"""
    devices = sd.query_devices()
    print("🎤 Доступные аудио устройства:")
    for i, device in enumerate(devices):
        if device['max_input_channels'] >= 2:
            print(f"   {i}: {device['name']} (Стерео: ✅)")
        else:
            print(f"   {i}: {device['name']} (Стерео: ❌)")
    
    # Проверяем устройство по умолчанию
    default_device = sd.default.device[0]
    if default_device is not None:
        device_info = sd.query_devices(default_device)
        if device_info['max_input_channels'] >= 2:
            print(f"\n✅ Устройство по умолчанию поддерживает стерео")
            return True
        else:
            print(f"\n⚠️ Устройство по умолчанию не поддерживает стерео")
            return False
    return False

def volume_display_thread():
    """Поток для отображения громкости и направления"""
    while is_running:
        if volume_buffer_left and volume_buffer_right:
            left_vol = volume_buffer_left[-1]
            right_vol = volume_buffer_right[-1]
            
            # Стерео индикатор громкости
            stereo_meter = create_stereo_volume_meter(left_vol, right_vol)
            
            # Отображение направления
            if direction_buffer:
                angle = direction_buffer[-1]
                direction_desc, emoji = get_direction_description(angle)
                direction_viz = create_direction_visualizer(angle)
                
                # Комбинированный вывод
                sys.stdout.write(f"\r🔊 {stereo_meter}")
                if len(direction_buffer) >= 3:
                    sys.stdout.write(f"\n🧭 {direction_desc} ({angle:.1f}°) {direction_viz}")
                sys.stdout.flush()
            
        time.sleep(0.2)

def audio_callback(indata, frames, time_info, status):
    """Обработка стерео аудио потока"""
    global last_prediction, last_time
    
    if status:
        print(f"\n⚠️ Audio status: {status}")
    
    try:
        # Разделяем на левый и правый каналы
        if indata.shape[1] >= 2:
            left_channel = indata[:, 0].astype(np.float32)
            right_channel = indata[:, 1].astype(np.float32)
        else:
            # Если моно, дублируем
            mono = indata[:, 0].astype(np.float32)
            left_channel = right_channel = mono
        
        # Вычисляем RMS для каждого канала
        left_rms = np.sqrt(np.mean(left_channel**2))
        right_rms = np.sqrt(np.mean(right_channel**2))
        avg_rms = (left_rms + right_rms) / 2
        
        # Добавляем в буферы
        volume_buffer_left.append(left_rms)
        volume_buffer_right.append(right_rms)
        
        # Вычисляем направление звука
        if avg_rms > RMS_THRESHOLD:
            angle, delay = calculate_direction(left_channel, right_channel)
            direction_buffer.append(angle)
        
        # Проверяем порог шума
        if avg_rms < RMS_THRESHOLD:
            return
        
        # Создаем моно для классификации (YAMNet работает с моно)
        mono_audio = (left_channel + right_channel) / 2
        
        # Классификация звука
        emb = extract_embedding_from_audio(mono_audio)
        pred = classifier.predict(np.expand_dims(emb, axis=0), verbose=0)[0]
        label_idx = np.argmax(pred)
        confidence = pred[label_idx]
        
        smoothed_prediction = smooth_predictions(label_idx)
        
        current_time = time.time()
        if (smoothed_prediction is not None and 
            confidence > MIN_CONFIDENCE and 
            smoothed_prediction != last_prediction and
            current_time - last_time > 1.5):
            
            label = class_names[smoothed_prediction]
            
            # Получаем информацию о направлении
            if direction_buffer:
                current_angle = direction_buffer[-1]
                direction_desc, emoji = get_direction_description(current_angle)
            else:
                current_angle = 0
                direction_desc, emoji = "📍 НЕИЗВЕСТНО", "❓"
            
            # Очищаем строку и выводим результат
            sys.stdout.write("\r" + " " * 100 + "\r")
            
            # Красивый вывод результата с направлением
            result_emoji = "🔴" if label == "negative" else "🟢"
            print(f"\n{result_emoji} === ОБНАРУЖЕНИЕ === {result_emoji}")
            print(f"🎧 Класс:       {label.upper()}")
            print(f"📊 Уверенность: {confidence:.2f} ({confidence*100:.1f}%)")
            print(f"🔊 Громкость:   L:{left_rms:.3f} R:{right_rms:.3f}")
            print(f"🧭 Направление: {direction_desc} ({current_angle:.1f}°)")
            print(f"{emoji} Источник звука")
            print(f"⏰ Время:       {time.strftime('%H:%M:%S')}")
            print("=" * 50)
            
            last_prediction = smoothed_prediction
            last_time = current_time
            
    except Exception as e:
        print(f"\n❌ Error in audio callback: {e}")

def main():
    print("🎙️ === REAL-TIME ДЕТЕКТОР С ОПРЕДЕЛЕНИЕМ НАПРАВЛЕНИЯ ===")
    print("🧭 Анализ направления звука с использованием стерео микрофона")
    print("=" * 60)
    
    # Проверяем аудио устройства
    if not check_audio_devices():
        print("\n⚠️ ВНИМАНИЕ: Для определения направления нужен стерео микрофон!")
        print("💡 Детектор будет работать, но направление определяться не будет.")
        input("Нажмите Enter для продолжения...")
    
    print(f"\n📊 Параметры:")
    print(f"   Минимальная уверенность: {MIN_CONFIDENCE}")
    print(f"   Сглаживание: {SMOOTH_WINDOW} кадров")
    print(f"   Порог шума: {RMS_THRESHOLD}")
    print(f"   Каналы: {CHANNELS} (стерео)")
    print(f"   Расстояние между микрофонами: {MIC_DISTANCE*100} см")
    print("=" * 60)
    print("🎯 Готов к работе! Говорите или издавайте звуки...")
    print("(Ctrl+C для выхода)\n")
    
    # Запускаем потоки
    volume_thread = threading.Thread(target=volume_display_thread, daemon=True)
    volume_thread.start()
    
    direction_thread = threading.Thread(target=display_direction_stats, daemon=True)
    direction_thread.start()
    
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
                sys.stdout.write("\r" + " " * 100 + "\r")
                print("🛑 Остановка детектора...")
                print("👋 До свидания!")
    except Exception as e:
        print(f"❌ Ошибка при инициализации аудио: {e}")
        print("💡 Убедитесь, что микрофон подключен и поддерживает стерео запись")

if __name__ == "__main__":
    main()