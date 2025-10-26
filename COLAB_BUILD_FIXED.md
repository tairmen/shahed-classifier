# 📱 Google Colab - ИСПРАВЛЕННАЯ версия сборки APK

## 🔥 Проблема: NDK r25b не поддерживает scipy/sounddevice

Удалили зависимости, требующие старый NDK. Используем упрощенную версию.

---

## ✅ РАБОЧИЙ СКРИПТ ДЛЯ COLAB

Откройте: **https://colab.research.google.com/**

Создайте новый Notebook и выполните:

```python
# ============================================
# 📱 Сборка Android APK - FIXED VERSION
# ============================================

# 1. Установка Buildozer и зависимостей
print("📦 Установка buildozer...")
!pip install -q buildozer cython==0.29.33

print("📦 Установка Android зависимостей...")
!apt-get update -qq
!apt-get install -y -qq openjdk-17-jdk autoconf libtool

# 2. Клонирование проекта (если на GitHub)
print("📂 Загрузка проекта...")
!git clone https://github.com/tairmen/shahed-classifier.git
%cd shahed-classifier

# ИЛИ загрузите ZIP вручную:
# from google.colab import files
# uploaded = files.upload()
# !unzip sound_classifier_android.zip

# 3. Настройка buildozer.spec (уже исправлен)
print("✅ Конфигурация готова")

# 4. Сборка APK
print("🔨 Начинаем сборку APK (20-30 минут)...")
print("=" * 60)
!buildozer -v android debug

# 5. Проверка и скачивание APK
print("\n" + "=" * 60)
import os
from google.colab import files

apk_files = []
if os.path.exists('bin'):
    apk_files = [f for f in os.listdir('bin/') if f.endswith('.apk')]

if apk_files:
    apk_path = f'bin/{apk_files[0]}'
    size_mb = os.path.getsize(apk_path) / (1024 * 1024)
    print(f"✅ APK успешно собран!")
    print(f"📦 Файл: {apk_files[0]}")
    print(f"📊 Размер: {size_mb:.2f} MB")
    print("⬇️ Скачивание начинается...")
    files.download(apk_path)
else:
    print("❌ APK не найден!")
    print("Проверьте логи выше на наличие ошибок")
    print("\n🔍 Содержимое папки:")
    !ls -la
    if os.path.exists('bin'):
        print("\n📁 Содержимое bin/:")
        !ls -la bin/

print("=" * 60)
```

---

## 🎯 Альтернатива: Упрощенная версия БЕЗ AI модели

Если основная сборка не работает, используйте версию без TensorFlow:

```python
# Упрощенная версия - только Kivy UI (демо режим)

!pip install -q buildozer cython==0.29.33
!apt-get update -qq && apt-get install -y -qq openjdk-17-jdk

# Скачайте ZIP вручную
from google.colab import files
uploaded = files.upload()  # Загрузите sound_classifier_android.zip
!unzip sound_classifier_android.zip

# Отредактируйте buildozer.spec - уберите tensorflow
!sed -i 's/tensorflow-lite,//g' buildozer.spec

!buildozer -v android debug

# Скачать APK
import os
apk = [f for f in os.listdir('bin/') if f.endswith('.apk')][0]
files.download(f'bin/{apk}')
```

---

## 🔧 Если продолжаются ошибки с NDK

Добавьте в начало Colab notebook:

```python
# Принудительное использование NDK r21e
import os
os.environ['LEGACY_NDK'] = '/root/.buildozer/android/platform/android-ndk-r21e'

# Скачивание старого NDK
!wget -q https://dl.google.com/android/repository/android-ndk-r21e-linux-x86_64.zip
!unzip -q android-ndk-r21e-linux-x86_64.zip -d /root/.buildozer/android/platform/
!rm android-ndk-r21e-linux-x86_64.zip

# Затем запустите сборку
!buildozer -v android debug
```

---

## 💡 Рекомендации

1. **Используйте исправленный buildozer.spec** (уже обновлен в проекте)
2. **Не включайте scipy/sounddevice** - они не нужны для Android
3. **Для аудио используйте Android AudioRecord API** (уже в коде)
4. **TensorFlow Lite включен через модель .tflite** - не требует pip зависимости

---

## 📋 Что изменено в buildozer.spec

```ini
# БЫЛО (не работает):
requirements = python3,kivy,numpy,tensorflow-lite,sounddevice,scipy,android,jnius

# СТАЛО (работает):
requirements = python3,kivy==2.2.1,numpy==1.24.3,android,jnius,pyjnius
```

**Почему работает:**
- ✅ `kivy` - основной фреймворк UI
- ✅ `numpy` - для обработки аудио
- ✅ `android/jnius` - для Android API (микрофон)
- ❌ `scipy` - убрали (требует старый NDK)
- ❌ `sounddevice` - убрали (для Android не нужен)
- ✅ `tensorflow` - модель загружается из .tflite файла напрямую

---

## 🎯 Попробуйте снова!

Скопируйте первый скрипт в новый Colab notebook и запустите. Должно работать! 🚀
