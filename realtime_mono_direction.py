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
CHANNELS = 1  # Моно микрофон

# Буферы для данных
prediction_buffer = deque(maxlen=SMOOTH_WINDOW)
volume_buffer = deque(maxlen=30)
frequency_buffer = deque(maxlen=20)
direction_estimate_buffer = deque(maxlen=10)
last_prediction = None
last_time = 0
is_running = True

class MonoDirectionEstimator:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.prev_audio = None
        self.movement_history = deque(maxlen=20)
        
    def estimate_direction_from_movement(self, audio_data):
        """
        Оценивает направление на основе изменения характеристик звука
        (приближение/удаление источника)
        """
        try:
            if self.prev_audio is None:
                self.prev_audio = audio_data.copy()
                return 0.0, "НЕПОДВИЖНО"
            
            # Анализ изменения громкости
            current_rms = np.sqrt(np.mean(audio_data**2))
            prev_rms = np.sqrt(np.mean(self.prev_audio**2))
            
            volume_change = current_rms - prev_rms
            
            # Анализ изменения спектра
            current_spectrum = np.abs(np.fft.fft(audio_data))[:len(audio_data)//2]
            prev_spectrum = np.abs(np.fft.fft(self.prev_audio))[:len(self.prev_audio)//2]
            
            # Центр тяжести спектра (показывает доминирующие частоты)
            freqs = np.fft.fftfreq(len(audio_data), 1/self.sample_rate)[:len(audio_data)//2]
            
            current_centroid = np.sum(freqs * current_spectrum) / (np.sum(current_spectrum) + 1e-10)
            prev_centroid = np.sum(freqs * prev_spectrum) / (np.sum(prev_spectrum) + 1e-10)
            
            centroid_change = current_centroid - prev_centroid
            
            # Эффект Доплера (приблизительный)
            doppler_estimate = centroid_change / prev_centroid if prev_centroid > 0 else 0
            
            # Комбинированная оценка направления
            if abs(volume_change) > 0.005:  # Значительное изменение
                if volume_change > 0:
                    direction_desc = "ПРИБЛИЖАЕТСЯ"
                    movement_score = min(1.0, volume_change * 100)
                else:
                    direction_desc = "УДАЛЯЕТСЯ"
                    movement_score = max(-1.0, volume_change * 100)
            else:
                direction_desc = "СТАБИЛЬНО"
                movement_score = 0.0
            
            self.movement_history.append(movement_score)
            self.prev_audio = audio_data.copy()
            
            return movement_score, direction_desc
            
        except Exception as e:
            return 0.0, "ОШИБКА"
    
    def analyze_frequency_distribution(self, audio_data):
        """
        Анализирует распределение частот для дополнительной информации
        """
        try:
            # FFT анализ
            fft = np.abs(np.fft.fft(audio_data))
            freqs = np.fft.fftfreq(len(audio_data), 1/self.sample_rate)
            
            # Положительные частоты
            positive_freqs = freqs[:len(freqs)//2]
            positive_fft = fft[:len(fft)//2]
            
            # Частотные полосы
            low_band = np.sum(positive_fft[(positive_freqs >= 80) & (positive_freqs < 300)])
            mid_band = np.sum(positive_fft[(positive_freqs >= 300) & (positive_freqs < 2000)])
            high_band = np.sum(positive_fft[(positive_freqs >= 2000) & (positive_freqs < 8000)])
            
            total_energy = low_band + mid_band + high_band
            
            if total_energy > 0:
                low_ratio = low_band / total_energy
                mid_ratio = mid_band / total_energy
                high_ratio = high_band / total_energy
                
                # Доминирующая частота
                dominant_freq_idx = np.argmax(positive_fft)
                dominant_freq = positive_freqs[dominant_freq_idx] if dominant_freq_idx < len(positive_freqs) else 0
                
                return {
                    'low_ratio': low_ratio,
                    'mid_ratio': mid_ratio,
                    'high_ratio': high_ratio,
                    'dominant_freq': dominant_freq,
                    'total_energy': total_energy
                }
            else:
                return None
                
        except Exception:
            return None

# Инициализируем оценщик направления
direction_estimator = MonoDirectionEstimator()

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

def create_movement_visualizer(movement_score, movement_desc, width=40):
    """Создает визуализацию движения источника звука"""
    # Нормализуем score от -1 до 1
    normalized = np.clip(movement_score, -1, 1)
    
    # Центр визуализации
    center = width // 2
    pos = center + int(normalized * (width // 4))
    pos = np.clip(pos, 0, width - 1)
    
    # Создаем визуализацию
    bar = list("░" * width)
    
    # Показываем направление движения
    if normalized > 0.2:  # Приближается
        for i in range(center, min(width, pos + 3)):
            bar[i] = "▶" if i == pos else "►"
    elif normalized < -0.2:  # Удаляется
        for i in range(max(0, pos - 2), center + 1):
            bar[i] = "◄" if i == pos else "◂"
    else:  # Стабильно
        bar[pos] = "⬤"
    
    # Метки
    bar[0] = "◄"
    bar[center] = "●"
    bar[width - 1] = "►"
    
    return f"|{(''.join(bar))}| {movement_desc}"

def create_frequency_analysis_display(freq_analysis):
    """Отображает анализ частот"""
    if freq_analysis is None:
        return "🎵 Нет данных о частотах"
    
    low = freq_analysis['low_ratio']
    mid = freq_analysis['mid_ratio'] 
    high = freq_analysis['high_ratio']
    dominant = freq_analysis['dominant_freq']
    
    # Визуальные полосы
    low_bar = "█" * int(low * 20)
    mid_bar = "█" * int(mid * 20)
    high_bar = "█" * int(high * 20)
    
    # Определяем тип звука
    if mid > 0.6:
        sound_type = "🗣️ РЕЧЬ/ГОЛОС"
    elif high > 0.5:
        sound_type = "🔔 ВЫСОКИЕ ЧАСТОТЫ"
    elif low > 0.5:
        sound_type = "🥁 НИЗКИЕ ЧАСТОТЫ"
    else:
        sound_type = "🎵 СМЕШАННЫЙ"
    
    result = f"""
🎵 === ЧАСТОТНЫЙ АНАЛИЗ ===
   Низкие (80-300 Hz):    |{low_bar:<20}| {low:.2f}
   Средние (300-2000 Hz): |{mid_bar:<20}| {mid:.2f}
   Высокие (2-8 kHz):     |{high_bar:<20}| {high:.2f}
   Доминирующая: {dominant:.0f} Hz
   Тип звука: {sound_type}
"""
    return result

def create_3d_sound_map(movement_score, freq_analysis, volume):
    """Создает 3D карту звука"""
    if freq_analysis is None:
        return "🗺️ Недостаточно данных для карты"
    
    # Определяем позицию источника
    if movement_score > 0.3:
        position = "🔊 ПРИБЛИЖАЮЩИЙСЯ ИСТОЧНИК"
        distance = "Близко"
    elif movement_score < -0.3:
        position = "🔇 УДАЛЯЮЩИЙСЯ ИСТОЧНИК"  
        distance = "Далеко"
    else:
        position = "📍 СТАЦИОНАРНЫЙ ИСТОЧНИК"
        distance = "Средне"
    
    # Оценка высоты источника по частотам
    if freq_analysis['high_ratio'] > 0.4:
        height = "⬆️ ВЫСОКО (много ВЧ)"
    elif freq_analysis['low_ratio'] > 0.4:
        height = "⬇️ НИЗКО (много НЧ)"
    else:
        height = "➡️ НА УРОВНЕ"
    
    # Интенсивность
    if volume > 0.05:
        intensity = "🔥 ГРОМКИЙ"
    elif volume > 0.02:
        intensity = "🔆 СРЕДНИЙ"
    else:
        intensity = "💡 ТИХИЙ"
    
    return f"""
🗺️ === 3D КАРТА ЗВУКА ===
   Позиция: {position}
   Расстояние: {distance}
   Высота: {height}
   Интенсивность: {intensity}
"""

def movement_stats_display():
    """Отображает статистику движения"""
    while is_running:
        if direction_estimate_buffer and frequency_buffer:
            recent_movements = list(direction_estimate_buffer)[-10:]
            recent_freqs = list(frequency_buffer)[-5:]
            
            if recent_movements and recent_freqs:
                avg_movement = np.mean(recent_movements)
                movement_stability = np.std(recent_movements)
                
                # Общая тенденция
                if avg_movement > 0.1:
                    trend = "📈 ИСТОЧНИК ПРИБЛИЖАЕТСЯ"
                elif avg_movement < -0.1:
                    trend = "📉 ИСТОЧНИК УДАЛЯЕТСЯ"
                else:
                    trend = "📊 ИСТОЧНИК СТАБИЛЕН"
                
                # Стабильность
                if movement_stability < 0.1:
                    stability = "🟢 СТАБИЛЬНО"
                elif movement_stability < 0.3:
                    stability = "🟡 УМЕРЕННО"
                else:
                    stability = "🔴 НЕСТАБИЛЬНО"
                
                latest_freq = recent_freqs[-1] if recent_freqs else None
                
                print(f"\n📊 === АНАЛИЗ ДВИЖЕНИЯ (МОНО МИКРОФОН) ===")
                print(f"   Тенденция: {trend}")
                print(f"   Средний сдвиг: {avg_movement:.3f}")
                print(f"   Стабильность: {stability}")
                
                if latest_freq:
                    print(create_frequency_analysis_display(latest_freq))
                    print(create_3d_sound_map(avg_movement, latest_freq, volume_buffer[-1] if volume_buffer else 0))
                
                print("=" * 60)
                
                for _ in range(3):
                    print()
        
        time.sleep(6)

def volume_display_thread():
    """Поток отображения состояния"""
    while is_running:
        if volume_buffer and direction_estimate_buffer:
            current_vol = volume_buffer[-1]
            movement_score = direction_estimate_buffer[-1] if direction_estimate_buffer else 0
            
            # Индикатор громкости
            vol_normalized = min(1.0, np.log10(current_vol * 1000 + 1) / 3) if current_vol > 0 else 0
            vol_bar = "█" * int(vol_normalized * 30) + "░" * (30 - int(vol_normalized * 30))
            
            # Движение
            movement_desc = "ПРИБЛИЖАЕТСЯ" if movement_score > 0.1 else "УДАЛЯЕТСЯ" if movement_score < -0.1 else "СТАБИЛЬНО"
            movement_viz = create_movement_visualizer(movement_score, movement_desc)
            
            # Комбинированный вывод
            sys.stdout.write(f"\r🔊 |{vol_bar}| {current_vol:.3f}")
            sys.stdout.write(f"\n🎯 {movement_viz}")
            sys.stdout.flush()
        
        time.sleep(0.3)

def audio_callback(indata, frames, time_info, status):
    """Обработка моно аудио с анализом движения"""
    global last_prediction, last_time
    
    if status:
        print(f"\n⚠️ Audio status: {status}")
    
    try:
        # Моно аудио
        mono = indata[:, 0].astype(np.float32)
        rms = np.sqrt(np.mean(mono**2))
        
        volume_buffer.append(rms)
        
        # Анализ движения и частот
        if rms > RMS_THRESHOLD:
            movement_score, movement_desc = direction_estimator.estimate_direction_from_movement(mono)
            freq_analysis = direction_estimator.analyze_frequency_distribution(mono)
            
            direction_estimate_buffer.append(movement_score)
            if freq_analysis:
                frequency_buffer.append(freq_analysis)
        
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
            current_time - last_time > 1.5):
            
            label = class_names[smoothed_prediction]
            
            # Информация о движении
            if direction_estimate_buffer and frequency_buffer:
                current_movement = direction_estimate_buffer[-1]
                movement_desc = "ПРИБЛИЖАЕТСЯ" if current_movement > 0.1 else "УДАЛЯЕТСЯ" if current_movement < -0.1 else "СТАБИЛЬНО"
                latest_freq = frequency_buffer[-1]
            else:
                current_movement = 0
                movement_desc = "НЕИЗВЕСТНО"
                latest_freq = None
            
            # Вывод результата
            sys.stdout.write("\r" + " " * 100 + "\r")
            
            result_emoji = "🔴" if label == "negative" else "🟢"
            print(f"\n{result_emoji} === ОБНАРУЖЕНИЕ (МОНО АНАЛИЗ) === {result_emoji}")
            print(f"🎧 Класс:         {label.upper()}")
            print(f"📊 Уверенность:   {confidence:.2f} ({confidence*100:.1f}%)")
            print(f"🔊 Громкость:     {rms:.3f}")
            print(f"🎯 Движение:      {movement_desc} ({current_movement:.3f})")
            
            if latest_freq:
                print(f"🎵 Доминирующая частота: {latest_freq['dominant_freq']:.0f} Hz")
                
                # Тип звука
                if latest_freq['mid_ratio'] > 0.6:
                    sound_type = "🗣️ РЕЧЬ/ГОЛОС"
                elif latest_freq['high_ratio'] > 0.5:
                    sound_type = "🔔 ВЫСОКИЕ ТОНА"
                elif latest_freq['low_ratio'] > 0.5:
                    sound_type = "🥁 НИЗКИЕ ТОНА"
                else:
                    sound_type = "🎵 СМЕШАННЫЙ"
                    
                print(f"🎼 Тип звука:     {sound_type}")
            
            print(f"⏰ Время:         {time.strftime('%H:%M:%S')}")
            
            # Визуализация движения
            movement_viz = create_movement_visualizer(current_movement, movement_desc)
            print(f"📍 Анализ:        {movement_viz}")
            
            print("=" * 60)
            
            last_prediction = smoothed_prediction
            last_time = current_time
            
    except Exception as e:
        print(f"\n❌ Error in audio callback: {e}")

def main():
    print("🎙️ === ДЕТЕКТОР ДЛЯ МОНО МИКРОФОНА ===")
    print("🎯 Анализ движения источника звука")
    print("🎵 Частотный анализ и классификация")
    print("=" * 50)
    
    # Проверяем аудио устройства
    devices = sd.query_devices()
    print("🎤 Доступные аудио устройства:")
    for i, device in enumerate(devices):
        if device['max_input_channels'] >= 1:
            print(f"   {i}: {device['name']} ✅")
    
    print(f"\n📊 Параметры:")
    print(f"   Режим: МОНО микрофон")
    print(f"   Анализ: Движение + Частоты")
    print(f"   Минимальная уверенность: {MIN_CONFIDENCE}")
    print(f"   Сглаживание: {SMOOTH_WINDOW} кадров")
    print(f"   Порог шума: {RMS_THRESHOLD}")
    print("=" * 50)
    print("🎯 Готов к анализу! Говорите или издавайте звуки...")
    print("💡 Двигайте источник звука для анализа движения")
    print("(Ctrl+C для выхода)\n")
    
    # Запускаем потоки
    volume_thread = threading.Thread(target=volume_display_thread, daemon=True)
    volume_thread.start()
    
    stats_thread = threading.Thread(target=movement_stats_display, daemon=True)
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
                sys.stdout.write("\r" + " " * 100 + "\r")
                print("🛑 Остановка детектора...")
                print("👋 До свидания!")
    except Exception as e:
        print(f"❌ Ошибка аудио: {e}")

if __name__ == "__main__":
    main()