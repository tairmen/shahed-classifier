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
MIN_CONFIDENCE = 0.8
SMOOTH_WINDOW = 3
RMS_THRESHOLD = 0.01

# Буферы для данных
prediction_buffer = deque(maxlen=SMOOTH_WINDOW)
volume_buffer = deque(maxlen=20)
last_prediction = None
last_time = 0
is_running = True

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

def create_ascii_meter(volume, width=40):
    """Создает ASCII индикатор громкости"""
    # Нормализация для отображения
    if volume <= 0:
        normalized = 0
    else:
        normalized = min(1.0, np.log10(volume * 1000 + 1) / 3)
    
    filled = int(normalized * width)
    bar = "█" * filled + "░" * (width - filled)
    
    # Индикатор статуса
    if volume < RMS_THRESHOLD:
        status = "🔇 ТИХО  "
    elif volume < 0.03:
        status = "🔉 СРЕДНЕ"
    else:
        status = "🔊 ГРОМКО"
    
    # Числовое значение
    return f"{status} |{bar}| {volume:.3f}"

def create_frequency_bars(audio_data, n_bars=20):
    """Создает простую частотную визуализацию"""
    if len(audio_data) < 512:
        return "🎵 " + "░" * n_bars
    
    # FFT для анализа частот
    fft = np.abs(np.fft.fft(audio_data[-512:]))[:256]
    
    # Группируем частоты в полосы
    band_size = len(fft) // n_bars
    bands = []
    
    for i in range(n_bars):
        start = i * band_size
        end = start + band_size
        band_power = np.mean(fft[start:end])
        bands.append(band_power)
    
    # Нормализуем и создаем визуализацию
    max_power = max(bands) if max(bands) > 0 else 1
    bars = ""
    
    for power in bands:
        normalized = power / max_power
        if normalized > 0.8:
            bars += "█"
        elif normalized > 0.6:
            bars += "▓"
        elif normalized > 0.4:
            bars += "▒"
        elif normalized > 0.2:
            bars += "░"
        else:
            bars += " "
    
    return f"🎵 |{bars}|"

def display_stats():
    """Отображает статистику в real-time"""
    stats_history = deque(maxlen=60)  # История за минуту
    
    while is_running:
        if volume_buffer:
            current_vol = volume_buffer[-1]
            avg_vol = np.mean(list(volume_buffer))
            max_vol = max(volume_buffer)
            
            stats_history.append(current_vol)
            
            # Создаем мини-график истории громкости
            if len(stats_history) >= 10:
                mini_graph = ""
                normalized_history = [v / (max(stats_history) or 1) for v in list(stats_history)[-20:]]
                for val in normalized_history:
                    if val > 0.8:
                        mini_graph += "▂"
                    elif val > 0.6:
                        mini_graph += "▁"
                    elif val > 0.4:
                        mini_graph += "▁"
                    else:
                        mini_graph += "_"
            else:
                mini_graph = "_" * 20
            
            # Статистики
            print(f"\n📊 === СТАТИСТИКА АУДИО ===")
            print(f"   Текущая громкость: {current_vol:.4f}")
            print(f"   Средняя громкость:  {avg_vol:.4f}")
            print(f"   Максимум:          {max_vol:.4f}")
            print(f"   История: {mini_graph}")
            print(f"   Порог шума:        {RMS_THRESHOLD:.4f}")
            print(f"   Статус: {'🟢 АКТИВЕН' if current_vol > RMS_THRESHOLD else '🔴 ТИШИНА'}")
            
            # Очищаем экран через некоторое время
            for _ in range(8):
                print()
        
        time.sleep(3)  # Обновляем каждые 3 секунды

def volume_display_thread():
    """Поток для отображения громкости"""
    current_audio = deque(maxlen=1024)
    
    while is_running:
        if volume_buffer:
            current_vol = volume_buffer[-1]
            
            # Основной индикатор громкости
            volume_meter = create_ascii_meter(current_vol)
            
            # Частотная визуализация (если есть аудио данные)
            if len(current_audio) > 100:
                freq_bars = create_frequency_bars(list(current_audio))
            else:
                freq_bars = "🎵 " + "░" * 20
            
            # Отображение
            sys.stdout.write(f"\r{volume_meter}")
            sys.stdout.flush()
            
            # Иногда показываем частотный анализ
            if int(time.time()) % 5 == 0:
                sys.stdout.write(f"\r{freq_bars}")
                sys.stdout.flush()
                time.sleep(0.5)
        
        time.sleep(0.1)

def audio_callback(indata, frames, time_info, status):
    global last_prediction, last_time
    
    if status:
        print(f"\n⚠️ Audio status: {status}")
    
    try:
        mono = np.mean(indata, axis=1).astype(np.float32)
        rms = np.sqrt(np.mean(mono**2))
        
        # Добавляем в буферы
        volume_buffer.append(rms)
        
        if rms < RMS_THRESHOLD:
            return
        
        # Классификация
        emb = extract_embedding_from_audio(mono)
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
            
            # Очищаем строку и выводим результат
            sys.stdout.write("\r" + " " * 80 + "\r")
            
            # Красивый вывод результата
            emoji = "🔴" if label == "negative" else "🟢"
            print(f"\n{emoji} === ОБНАРУЖЕНИЕ === {emoji}")
            print(f"🎧 Класс:       {label.upper()}")
            print(f"📊 Уверенность: {confidence:.2f} ({confidence*100:.1f}%)")
            print(f"🔊 Громкость:   {rms:.3f}")
            print(f"⏰ Время:       {time.strftime('%H:%M:%S')}")
            print("=" * 40)
            
            last_prediction = smoothed_prediction
            last_time = current_time
            
    except Exception as e:
        print(f"\n❌ Error in audio callback: {e}")

def main():
    print("🎙️ === ПРОДВИНУТЫЙ REAL-TIME ДЕТЕКТОР ===")
    print(f"📊 Минимальная уверенность: {MIN_CONFIDENCE}")
    print(f"🔄 Сглаживание: {SMOOTH_WINDOW} кадров")
    print(f"🎚️ Порог шума: {RMS_THRESHOLD}")
    print(f"🎵 Частота дискретизации: {SAMPLE_RATE} Hz")
    print("📈 Включена визуализация громкости и частот")
    print("=" * 50)
    print("🎯 Готов к работе! Говорите или издавайте звуки...")
    print("(Ctrl+C для выхода)\n")
    
    # Запускаем поток визуализации
    volume_thread = threading.Thread(target=volume_display_thread, daemon=True)
    volume_thread.start()
    
    # Запускаем поток статистики
    stats_thread = threading.Thread(target=display_stats, daemon=True)
    stats_thread.start()
    
    # Основной аудио поток
    with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, 
                      callback=audio_callback, blocksize=BLOCK_SIZE):
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            global is_running
            is_running = False
            sys.stdout.write("\r" + " " * 80 + "\r")
            print("🛑 Остановка детектора...")
            print("👋 До свидания!")

if __name__ == "__main__":
    main()