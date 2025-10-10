import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import sounddevice as sd
import time
from collections import deque
import threading
import sys

# Пути
MODEL_PATH = "model/my_sound_model.h5"
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
classifier = tf.keras.models.load_model(MODEL_PATH)
class_names = ["negative", "positive"]

SAMPLE_RATE = 16000
BLOCK_DURATION = 1.0
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DURATION)
MIN_CONFIDENCE = 0.8  # Минимальная уверенность для вывода
SMOOTH_WINDOW = 3     # Количество последних предсказаний для сглаживания
RMS_THRESHOLD = 0.01  # Порог шума

# Буферы для сглаживания предсказаний и громкости
prediction_buffer = deque(maxlen=SMOOTH_WINDOW)
volume_buffer = deque(maxlen=10)  # Для сглаживания отображения громкости
last_prediction = None
last_time = 0
current_volume = 0.0
is_running = True

def extract_embedding_from_audio(audio):
    scores, embeddings, spectrogram = yamnet_model(audio)
    embedding = tf.reduce_mean(embeddings, axis=0)
    return embedding

def smooth_predictions(new_prediction):
    """Сглаживает предсказания для уменьшения ложных срабатываний"""
    prediction_buffer.append(new_prediction)
    
    if len(prediction_buffer) < SMOOTH_WINDOW:
        return None
    
    # Используем мажоритарное голосование
    negative_count = sum(1 for p in prediction_buffer if p == 0)
    positive_count = sum(1 for p in prediction_buffer if p == 1)
    
    return 0 if negative_count >= positive_count else 1

def create_volume_bar(volume, max_width=40):
    """Создает визуальную полоску громкости"""
    # Нормализуем громкость для отображения (логарифмическая шкала)
    if volume <= 0:
        normalized = 0
    else:
        # Логарифмическая шкала для лучшего отображения
        normalized = min(1.0, np.log10(volume * 1000 + 1) / 3)
    
    filled_width = int(normalized * max_width)
    bar = "█" * filled_width + "░" * (max_width - filled_width)
    
    # Цветовая кодировка
    if volume < RMS_THRESHOLD:
        color = "🔇"  # Тихо
    elif volume < 0.05:
        color = "🔉"  # Средне
    else:
        color = "🔊"  # Громко
    
    return f"{color} |{bar}| {volume:.3f}"

def volume_display_thread():
    """Отдельный поток для отображения громкости в real-time"""
    while is_running:
        if volume_buffer:
            avg_volume = np.mean(list(volume_buffer))
            volume_bar = create_volume_bar(avg_volume)
            
            # Очищаем строку и выводим громкость
            sys.stdout.write(f"\r{volume_bar}")
            sys.stdout.flush()
        
        time.sleep(0.1)  # Обновляем 10 раз в секунду

def audio_callback(indata, frames, time_info, status):
    global last_prediction, last_time, current_volume
    
    if status:
        print(f"\n⚠️ Audio status: {status}")
    
    try:
        # Вычисляем уровень громкости (RMS)
        mono = np.mean(indata, axis=1).astype(np.float32)
        rms = np.sqrt(np.mean(mono**2))
        
        # Добавляем в буфер для сглаживания отображения
        volume_buffer.append(rms)
        current_volume = rms
        
        # Игнорируем слишком тихие звуки (фоновый шум)
        if rms < RMS_THRESHOLD:
            return
        
        # Извлекаем эмбеддинг и предсказываем
        emb = extract_embedding_from_audio(mono)
        pred = classifier.predict(np.expand_dims(emb, axis=0), verbose=0)[0]
        label_idx = np.argmax(pred)
        confidence = pred[label_idx]
        
        # Применяем сглаживание
        smoothed_prediction = smooth_predictions(label_idx)
        
        # Выводим только если есть высокая уверенность и изменение
        current_time = time.time()
        if (smoothed_prediction is not None and 
            confidence > MIN_CONFIDENCE and 
            smoothed_prediction != last_prediction and
            current_time - last_time > 1.0):  # Минимум 1 секунда между выводами
            
            label = class_names[smoothed_prediction]
            # Очищаем строку громкости и выводим результат
            sys.stdout.write("\r" + " " * 60 + "\r")  # Очищаем строку
            print(f"🎧 {label.upper()} (conf: {confidence:.2f}, vol: {rms:.3f})")
            last_prediction = smoothed_prediction
            last_time = current_time
            
    except Exception as e:
        print(f"\n❌ Error in audio callback: {e}")

print("🎙️ Улучшенный детектор с визуализацией громкости")
print(f"📊 Минимальная уверенность: {MIN_CONFIDENCE}")
print(f"🔄 Сглаживание: {SMOOTH_WINDOW} кадров")
print(f"🎚️ Порог шума: {RMS_THRESHOLD}")
print("📈 Визуализация громкости в real-time")
print("(Ctrl+C для выхода)\n")

# Запускаем поток для отображения громкости
volume_thread = threading.Thread(target=volume_display_thread, daemon=True)
volume_thread.start()

with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, callback=audio_callback, blocksize=BLOCK_SIZE):
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        is_running = False
        sys.stdout.write("\r" + " " * 60 + "\r")  # Очищаем строку
        print("🛑 Остановка детектора")
