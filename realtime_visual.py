import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import sounddevice as sd
import time
from collections import deque
import threading
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

# Пути
MODEL_PATH = "model/my_sound_model.h5"
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
classifier = tf.keras.models.load_model(MODEL_PATH)
class_names = ["negative", "positive"]

SAMPLE_RATE = 16000
BLOCK_DURATION = 0.5
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DURATION)
MIN_CONFIDENCE = 0.8
SMOOTH_WINDOW = 3
RMS_THRESHOLD = 0.01

# Буферы данных
prediction_buffer = deque(maxlen=SMOOTH_WINDOW)
audio_buffer = deque(maxlen=100)  # История аудио для спектрограммы
volume_history = deque(maxlen=100)  # История громкости
prediction_history = deque(maxlen=50)  # История предсказаний

last_prediction = None
last_time = 0
is_running = True
current_label = "ТИШИНА"
current_confidence = 0.0

class AudioVisualizer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🎙️ Real-time Audio Classifier")
        self.root.geometry("1200x800")
        self.root.configure(bg='black')
        
        # Создаем фигуру matplotlib
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(12, 8))
        self.fig.patch.set_facecolor('black')
        
        # Настройка осей
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.set_facecolor('black')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
        
        # График спектрограммы
        self.ax1.set_title('📊 Спектрограмма (Real-time)', color='white', fontsize=14)
        self.ax1.set_ylabel('Частота (Hz)', color='white')
        
        # График громкости
        self.ax2.set_title('🔊 Уровень громкости', color='white', fontsize=14)
        self.ax2.set_ylabel('RMS', color='white')
        self.ax2.set_ylim(0, 0.1)
        
        # График предсказаний
        self.ax3.set_title('🎧 История классификации', color='white', fontsize=14)
        self.ax3.set_ylabel('Класс', color='white')
        self.ax3.set_xlabel('Время', color='white')
        self.ax3.set_ylim(-0.5, 1.5)
        self.ax3.set_yticks([0, 1])
        self.ax3.set_yticklabels(['NEGATIVE', 'POSITIVE'])
        
        # Встраиваем matplotlib в tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Статус панель
        self.status_frame = tk.Frame(self.root, bg='black', height=100)
        self.status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_label = tk.Label(
            self.status_frame, 
            text="🎙️ Инициализация...", 
            font=("Arial", 16, "bold"),
            fg='white', 
            bg='black'
        )
        self.status_label.pack(pady=10)
        
        # Запускаем анимацию
        self.ani = animation.FuncAnimation(
            self.fig, self.update_plots, interval=100, blit=False
        )
    
    def update_plots(self, frame):
        # Очищаем оси
        self.ax1.clear()
        self.ax2.clear() 
        self.ax3.clear()
        
        # Обновляем настройки осей
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.set_facecolor('black')
            ax.tick_params(colors='white')
        
        # График спектрограммы
        if len(audio_buffer) > 0:
            recent_audio = list(audio_buffer)[-10:]  # Последние 10 блоков
            if recent_audio:
                # Создаем спектрограмму
                combined_audio = np.concatenate(recent_audio)
                if len(combined_audio) > 1024:
                    # FFT для спектрограммы
                    freqs = np.fft.fftfreq(1024, 1/SAMPLE_RATE)[:512]
                    magnitude = np.abs(np.fft.fft(combined_audio[-1024:]))[:512]
                    
                    self.ax1.plot(freqs, magnitude, color='cyan', linewidth=1)
                    self.ax1.set_xlim(0, 8000)  # До 8kHz
                    self.ax1.fill_between(freqs, magnitude, alpha=0.3, color='cyan')
        
        self.ax1.set_title('📊 Спектрограмма (Real-time)', color='white', fontsize=12)
        self.ax1.set_ylabel('Амплитуда', color='white')
        
        # График громкости
        if len(volume_history) > 0:
            times = list(range(len(volume_history)))
            volumes = list(volume_history)
            
            self.ax2.plot(times, volumes, color='green', linewidth=2)
            self.ax2.fill_between(times, volumes, alpha=0.3, color='green')
            self.ax2.axhline(y=RMS_THRESHOLD, color='red', linestyle='--', alpha=0.7)
            
        self.ax2.set_title('🔊 Уровень громкости', color='white', fontsize=12)
        self.ax2.set_ylabel('RMS', color='white')
        self.ax2.set_ylim(0, max(0.1, max(volume_history) if volume_history else 0.1))
        
        # График предсказаний
        if len(prediction_history) > 0:
            times = list(range(len(prediction_history)))
            predictions = list(prediction_history)
            
            colors = ['red' if p == 0 else 'lime' for p in predictions]
            self.ax3.scatter(times, predictions, c=colors, s=50, alpha=0.8)
            
        self.ax3.set_title('🎧 История классификации', color='white', fontsize=12)
        self.ax3.set_ylabel('Класс', color='white')
        self.ax3.set_ylim(-0.5, 1.5)
        self.ax3.set_yticks([0, 1])
        self.ax3.set_yticklabels(['NEGATIVE', 'POSITIVE'])
        
        # Обновляем статус
        current_volume = volume_history[-1] if volume_history else 0
        volume_bar = "█" * int(current_volume * 1000) + "░" * (20 - int(current_volume * 1000))
        
        status_text = f"🎧 {current_label} | Уверенность: {current_confidence:.2f} | Громкость: |{volume_bar}| {current_volume:.3f}"
        self.status_label.config(text=status_text)
        
        return []
    
    def run(self):
        self.root.mainloop()

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

def audio_callback(indata, frames, time_info, status):
    global last_prediction, last_time, current_label, current_confidence
    
    if status:
        print(f"⚠️ Audio status: {status}")
    
    try:
        mono = np.mean(indata, axis=1).astype(np.float32)
        rms = np.sqrt(np.mean(mono**2))
        
        # Добавляем в буферы для визуализации
        audio_buffer.append(mono.copy())
        volume_history.append(rms)
        
        if rms < RMS_THRESHOLD:
            current_label = "ТИШИНА"
            current_confidence = 0.0
            return
        
        # Обработка звука
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
            
            current_label = class_names[smoothed_prediction].upper()
            current_confidence = confidence
            prediction_history.append(smoothed_prediction)
            last_prediction = smoothed_prediction
            last_time = current_time
            
            print(f"🎧 {current_label} (conf: {confidence:.2f}, vol: {rms:.3f})")
            
    except Exception as e:
        print(f"❌ Error in audio callback: {e}")

def main():
    print("🎙️ Запуск расширенного real-time детектора с визуализацией...")
    print("📊 Загрузка интерфейса...")
    
    # Создаем и запускаем визуализатор в отдельном потоке
    visualizer = AudioVisualizer()
    
    def start_audio():
        with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, 
                          callback=audio_callback, blocksize=BLOCK_SIZE):
            try:
                while is_running:
                    time.sleep(0.1)
            except Exception as e:
                print(f"Ошибка аудио потока: {e}")
    
    # Запускаем аудио в отдельном потоке
    audio_thread = threading.Thread(target=start_audio, daemon=True)
    audio_thread.start()
    
    # Запускаем GUI
    visualizer.run()

if __name__ == "__main__":
    main()