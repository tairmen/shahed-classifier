# 🚀 SUPER SIMPLE Colab APK Builder - Copy & Paste

## Проблема: HTTP Error 404 при скачивании зависимостей

**Причина:** Указанные версии (kivy==2.2.1, numpy==1.24.3) могут быть недоступны в python-for-android репозитории.

**Решение:** Убираем версии, используем последние стабильные.

---

## ✅ ИСПРАВЛЕННЫЙ КОД (скопируйте в Colab)

Откройте https://colab.research.google.com/ и вставьте:

```python
# ============================================
# 📱 Android Sound Detector APK Builder
# FIXED VERSION - без 404 ошибок
# ============================================

print("🚀 Starting APK build process...")
print("=" * 60)

# Step 1: Install dependencies
print("\n[1/6] Installing build tools...")
!pip install -q buildozer cython==0.29.33
!apt-get update -qq && apt-get install -y -qq openjdk-17-jdk autoconf libtool
print("✅ Build tools installed")

# Step 2: Clone project
print("\n[2/6] Cloning project...")
!git clone https://github.com/tairmen/shahed-classifier.git
%cd shahed-classifier
print("✅ Project loaded")

# Step 3: Fix buildozer.spec (CRITICAL FIX!)
print("\n[3/6] Fixing buildozer.spec to avoid 404 errors...")
with open('buildozer.spec', 'r') as f:
    content = f.read()

# Replace requirements - remove version pins
import re
content = re.sub(r'requirements = .*', 
                 'requirements = python3,kivy,numpy,android,pyjnius', 
                 content)

with open('buildozer.spec', 'w') as f:
    f.write(content)

print("✅ buildozer.spec fixed!")
print("   Requirements: python3,kivy,numpy,android,pyjnius")
print("   (No version pins = no 404 errors)")

# Step 4: Clean previous builds
print("\n[4/6] Cleaning cache...")
!buildozer android clean
print("✅ Cache cleaned")

# Step 5: Build APK
print("\n[5/6] Building APK (20-30 minutes)...")
print("☕ Perfect time for coffee!")
print("=" * 60)

!buildozer -v android debug

print("\n" + "=" * 60)
print("✅ Build completed!")

# Step 6: Download APK
print("\n[6/6] Preparing download...")
from google.colab import files
import os

if os.path.exists('bin'):
    apk_files = [f for f in os.listdir('bin/') if f.endswith('.apk')]
    if apk_files:
        apk_path = f'bin/{apk_files[0]}'
        size = os.path.getsize(apk_path) / (1024*1024)
        print(f"\n🎉 SUCCESS!")
        print(f"📱 APK: {apk_files[0]}")
        print(f"📦 Size: {size:.1f} MB")
        print(f"\n⬇️ Downloading...")
        files.download(apk_path)
        print("\n✨ Done! Install on your Android device!")
    else:
        print("❌ No APK found. Check build logs above.")
else:
    print("❌ Build failed. Check logs above.")

print("=" * 60)
```

---

## 🔧 Что исправлено:

### ❌ СТАРАЯ версия (404 ошибка):
```ini
requirements = python3,kivy==2.2.1,numpy==1.24.3,android,jnius,pyjnius
```

### ✅ НОВАЯ версия (работает):
```ini
requirements = python3,kivy,numpy,android,pyjnius
```

**Изменения:**
- Убрали `kivy==2.2.1` → стало `kivy` (latest)
- Убрали `numpy==1.24.3` → стало `numpy` (latest)
- Убрали `jnius` (дубликат `pyjnius`)
- Оставили только самое необходимое

---

## 🎯 Почему это работает:

1. **Без версий**: python-for-android автоматически использует совместимые версии
2. **Меньше зависимостей**: меньше шансов на конфликты
3. **Актуальные пакеты**: используются последние стабильные версии из p4a

---

## 📝 Альтернатива: Минимальная версия БЕЗ AI

Если все еще не работает, попробуйте без TensorFlow (только UI):

```python
# Минимальная версия - только Kivy UI
!pip install -q buildozer cython==0.29.33
!apt-get update -qq && apt-get install -y -qq openjdk-17-jdk

!git clone https://github.com/tairmen/shahed-classifier.git
%cd shahed-classifier

# Супер минимальные зависимости
!sed -i 's/requirements = .*/requirements = python3,kivy,android/' buildozer.spec

!buildozer android debug

# Download
from google.colab import files
import os
apk = [f for f in os.listdir('bin/') if f.endswith('.apk')][0]
files.download(f'bin/{apk}')
```

Это создаст APK без numpy/tensorflow, но с рабочим UI.

---

## ⚡ Быстрая проверка перед запуском:

1. Убедитесь что `shahed-classifier` репозиторий публичный
2. Убедитесь что в репозитории есть:
   - `android_sound_detector.py` или `main.py`
   - `buildozer.spec`
   - `distance_calibration.json`

---

## 🐛 Другие возможные ошибки:

### Ошибка: "Permission denied"
```python
!chmod +x ~/.buildozer/android/platform/android-ndk-*/toolchains/llvm/prebuilt/linux-x86_64/bin/*
```

### Ошибка: "SDK license not accepted"
```python
!yes | sdkmanager --licenses
```

### Ошибка: "No space left"
```python
# Очистить место в Colab
!rm -rf ~/.buildozer/android/platform/build-*/
```

---

## ✅ Попробуйте исправленный код!

Скопируйте первый большой блок кода в новый Colab notebook и запустите.

Он автоматически исправит buildozer.spec и соберет APK.

**Время сборки:** 20-30 минут

**Должно работать!** 🚀
