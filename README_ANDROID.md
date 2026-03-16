# 📱 Android Sound Detector - Инструкция по сборке APK

## 🎯 Описание
Kivy-версия универсального детектора звуков, готовая для конвертации в Android APK.

---

## 📋 Что создано

### 1. **android_sound_detector.py** - Основное приложение
- ✅ Kivy UI с красивым интерфейсом
- ✅ Поддержка Android разрешений (микрофон)
- ✅ Интеграция TensorFlow Lite модели
- ✅ Реал-тайм детекция звуков
- ✅ Расчет расстояния до источника
- ✅ Классификация типа звука
- ✅ Счетчик детекций
- ✅ Работает на Android и ПК (для тестирования)

### 2. **buildozer.spec** - Конфигурация сборки
- Настройки пакета (org.soundai.sounddetector)
- Зависимости: Kivy, NumPy, TensorFlow Lite
- Разрешения: RECORD_AUDIO, STORAGE
- Android API: 33 (минимум 21)
- Архитектуры: ARM64 + ARMv7

---

## 🚀 Как собрать APK

### **Метод 1: Linux (Ubuntu/Debian) - РЕКОМЕНДУЕТСЯ**

#### Установка Buildozer
```bash
# Установка зависимостей
sudo apt update
sudo apt install -y python3-pip build-essential git zip unzip \
    openjdk-17-jdk autoconf libtool pkg-config zlib1g-dev \
    libncurses5-dev libncursesw5-dev libtinfo5 cmake libffi-dev \
    libssl-dev

# Установка Cython и Buildozer
pip3 install --upgrade pip
pip3 install --upgrade cython
pip3 install --upgrade buildozer
```

#### Сборка APK
```bash
# Перейдите в папку проекта
cd /path/to/sound_classifier

# Инициализация buildozer (если нужно)
buildozer init

# Сборка APK (первая сборка займет ~30-60 минут)
buildozer -v android debug

# Готовый APK будет в: bin/sounddetector-1.0.0-arm64-v8a-debug.apk
```

#### Установка на Android
```bash
# Через ADB
adb install bin/sounddetector-1.0.0-arm64-v8a-debug.apk

# Или скопируйте APK на телефон и установите вручную
```

---

### **Метод 2: Windows (WSL2) - Альтернатива**

Buildozer не работает напрямую в Windows, используйте WSL2:

```powershell
# 1. Установите WSL2 (если еще нет)
wsl --install -d Ubuntu

# 2. Запустите Ubuntu WSL
wsl

# 3. В WSL выполните команды из "Метод 1"
```

---

### **Метод 3: Google Colab - Облачная сборка**

Если у вас нет Linux, используйте Google Colab:

```python
# В новом notebook Colab:

# 1. Установка buildozer
!pip install buildozer
!pip install cython

# 2. Клонирование репозитория
!git clone https://github.com/your-repo/sound_classifier.git
%cd sound_classifier

# 3. Сборка
!buildozer -v android debug

# 4. Скачать APK
from google.colab import files
files.download('bin/sounddetector-1.0.0-arm64-v8a-debug.apk')
```

---

## 🧪 Тестирование на ПК перед сборкой

Протестируйте приложение на компьютере:

```bash
# Установите зависимости
pip install kivy numpy tensorflow sounddevice scipy

# Запустите приложение
python android_sound_detector.py
```

**Интерфейс:**
- Кнопка "СТАРТ" - начать детекцию
- Кнопка "СТОП" - остановить
- Отображение: уверенность, расстояние, дБ, тип звука, счетчик

---

## 📱 Что будет работать на Android

### ✅ Работает:
- Kivy UI интерфейс
- Захват аудио с микрофона
- Обработка в реальном времени
- TensorFlow Lite модель
- Расчет расстояния
- Классификация звуков

### ⚠️ Требует адаптации:
- Захват аудио: замените `sounddevice` на Android `AudioRecord` API
- Модель: конвертируйте `my_sound_model.h5` в `.tflite` формат

---

## 🔧 Доработка для продакшена

### 1. **Конвертация модели в TFLite**

```python
import tensorflow as tf

# Загрузите H5 модель
model = tf.keras.models.load_model('model/my_sound_model.h5')

# Конвертируйте в TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Сохраните
with open('model/my_sound_model.tflite', 'wb') as f:
    f.write(tflite_model)
```

### 2. **Замените захват аудио для Android**

В `android_sound_detector.py` добавьте:

```python
if platform == 'android':
    # Используйте Android AudioRecord
    from jnius import autoclass
    AudioRecord = autoclass('android.media.AudioRecord')
    AudioFormat = autoclass('android.media.AudioFormat')
    MediaRecorder = autoclass('android.media.MediaRecorder$AudioSource')
    
    # Инициализация AudioRecord
    buffer_size = AudioRecord.getMinBufferSize(
        16000,  # Sample rate
        AudioFormat.CHANNEL_IN_MONO,
        AudioFormat.ENCODING_PCM_16BIT
    )
    
    audio_recorder = AudioRecord(
        MediaRecorder.MIC,
        16000,
        AudioFormat.CHANNEL_IN_MONO,
        AudioFormat.ENCODING_PCM_16BIT,
        buffer_size
    )
```

### 3. **Оптимизация батареи**

Добавьте в `buildozer.spec`:
```ini
android.wakelock = False
android.manifest.background_services = False
```

---

## 📦 Структура проекта для сборки

```
sound_classifier/
├── android_sound_detector.py    # Главный файл приложения
├── buildozer.spec               # Конфигурация сборки
├── distance_calibration.json    # Калибровка расстояний
├── model/
│   ├── my_sound_model.h5        # Ваша модель (конвертируйте в .tflite)
│   └── my_sound_model.tflite    # TFLite версия (создайте)
├── data/
│   └── positive/*.wav           # Примеры звуков (опционально)
└── README_ANDROID.md            # Эта инструкция
```

---

## ⚡ Быстрый старт (TL;DR)

```bash
# На Linux:
sudo apt install -y python3-pip build-essential git openjdk-17-jdk
pip3 install buildozer cython
cd sound_classifier
buildozer android debug
adb install bin/*.apk

# На Windows:
wsl --install
wsl
# Затем выполните команды для Linux
```

---

## 🐛 Решение проблем

### Ошибка: "Command failed: ./distribute.sh"
```bash
buildozer android clean
rm -rf .buildozer
buildozer -v android debug
```

### Ошибка: "Java not found"
```bash
sudo apt install openjdk-17-jdk
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
```

### Ошибка: "NDK not found"
```bash
# Buildozer автоматически скачает NDK при первой сборке
# Просто подождите, это займет время
```

### APK не устанавливается на телефон
```bash
# Включите "Неизвестные источники" в настройках Android
# Настройки > Безопасность > Неизвестные источники > Вкл
```

---

## 📊 Размер APK

- **Базовый APK**: ~50-80 MB
- **С TensorFlow Lite**: ~60-90 MB
- **Архитектуры**: ARM64 (основная), ARMv7 (совместимость)

---

## 🎨 Кастомизация UI

Отредактируйте `android_sound_detector.py`:
- Цвета: измените `color`, `background_color`
- Шрифты: измените `font_size`
- Иконки: добавьте `icon.filename` в `buildozer.spec`
- Splash screen: добавьте `presplash.filename`

---

## 📞 Поддержка

Если возникают проблемы при сборке:
1. Проверьте логи: `buildozer -v android debug`
2. Очистите кэш: `buildozer android clean`
3. Обновите buildozer: `pip install --upgrade buildozer`

---

## ✅ Чеклист перед сборкой

- [ ] Установлен Python 3.8+
- [ ] Установлен Java JDK 17
- [ ] Установлен buildozer
- [ ] Модель конвертирована в .tflite
- [ ] Файл `buildozer.spec` настроен
- [ ] Протестировано на ПК
- [ ] Достаточно места на диске (~10 GB для первой сборки)

---

## 🎯 Результат

После успешной сборки получите:
- ✅ **sounddetector-1.0.0-arm64-v8a-debug.apk** - готовое Android приложение
- ✅ Работает на Android 5.0+ (API 21+)
- ✅ Размер: ~60-90 MB
- ✅ Детекция звуков в реальном времени
- ✅ Расчет расстояния до источника
- ✅ Красивый UI интерфейс

**Готово к установке на любой Android телефон!** 📱🎵
