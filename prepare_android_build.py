"""
🚀 Автоматический скрипт подготовки проекта для сборки APK
Запускайте перед отправкой на сборку (WSL/Colab/Linux)
"""

import os
import shutil
import json
from pathlib import Path

def prepare_for_android_build():
    """Подготовка проекта к сборке Android APK"""
    
    print("=" * 60)
    print("📱 Подготовка проекта для сборки Android APK")
    print("=" * 60)
    
    # 1. Проверка необходимых файлов
    print("\n1️⃣ Проверка необходимых файлов...")
    required_files = {
        'android_sound_detector.py': 'Главное приложение',
        'buildozer.spec': 'Конфигурация сборки',
        'distance_calibration.json': 'Калибровка расстояний',
        'model/my_sound_model.h5': 'AI модель (H5)',
        'model/my_sound_model.tflite': 'AI модель (TFLite)',
    }
    
    missing = []
    for file, desc in required_files.items():
        if os.path.exists(file):
            print(f"   ✅ {file} - {desc}")
        else:
            print(f"   ❌ {file} - {desc} НЕ НАЙДЕН!")
            missing.append(file)
    
    if missing:
        print(f"\n⚠️ Отсутствуют файлы: {', '.join(missing)}")
        if 'model/my_sound_model.tflite' in missing:
            print("   Запустите: python convert_to_tflite.py")
        return False
    
    # 2. Переименование главного файла для Buildozer
    print("\n2️⃣ Подготовка главного файла...")
    
    # Buildozer ожидает main.py, но мы используем android_sound_detector.py
    # Создадим копию как main.py
    if not os.path.exists('main.py'):
        shutil.copy('android_sound_detector.py', 'main.py')
        print("   ✅ Создан main.py из android_sound_detector.py")
    else:
        print("   ℹ️ main.py уже существует")
    
    # 3. Проверка buildozer.spec
    print("\n3️⃣ Проверка buildozer.spec...")
    with open('buildozer.spec', 'r', encoding='utf-8') as f:
        spec_content = f.read()
    
    checks = {
        'title = Sound Detector': 'Название приложения',
        'package.name = sounddetector': 'Имя пакета',
        'RECORD_AUDIO': 'Разрешение микрофона',
        'tensorflow-lite': 'TensorFlow Lite',
    }
    
    for check, desc in checks.items():
        if check in spec_content:
            print(f"   ✅ {desc}")
        else:
            print(f"   ⚠️ {desc} - не найден в конфигурации")
    
    # 4. Создание requirements.txt для buildozer
    print("\n4️⃣ Создание requirements.txt...")
    requirements = [
        'kivy==2.2.1',
        'numpy==1.24.3',
        'tensorflow-lite==2.13.0',
        'pyjnius==1.5.0',
    ]
    
    with open('requirements_android.txt', 'w') as f:
        f.write('\n'.join(requirements))
    
    print(f"   ✅ Создан requirements_android.txt")
    
    # 5. Проверка размеров файлов
    print("\n5️⃣ Проверка размеров файлов...")
    
    h5_size = os.path.getsize('model/my_sound_model.h5') / (1024 * 1024)
    tflite_size = os.path.getsize('model/my_sound_model.tflite') / (1024 * 1024)
    
    print(f"   📦 H5 модель: {h5_size:.2f} MB")
    print(f"   📦 TFLite модель: {tflite_size:.2f} MB")
    
    if tflite_size > 50:
        print(f"   ⚠️ TFLite модель большая! Рекомендуется квантизация.")
    else:
        print(f"   ✅ Размер TFLite модели оптимален")
    
    # 6. Подсчет аудио файлов
    print("\n6️⃣ Подсчет данных...")
    
    positive_count = len(list(Path('data/positive').glob('*.wav'))) if os.path.exists('data/positive') else 0
    negative_count = len(list(Path('data/negative').glob('*.wav'))) if os.path.exists('data/negative') else 0
    
    print(f"   📊 Positive примеров: {positive_count}")
    print(f"   📊 Negative примеров: {negative_count}")
    print(f"   ℹ️ Эти файлы НЕ включаются в APK (только модель)")
    
    # 7. Создание скрипта для WSL
    print("\n7️⃣ Создание WSL скрипта...")
    
    wsl_script = """#!/bin/bash
# Automated APK build in WSL

echo "=========================================="
echo "Building Android APK in WSL"
echo "=========================================="

# Install dependencies (if not already installed)
echo "Step 1: Checking dependencies..."
if ! command -v buildozer &> /dev/null; then
    echo "Installing buildozer..."
    sudo apt update
    sudo apt install -y python3-pip build-essential git openjdk-17-jdk
    pip3 install --upgrade cython==0.29.33 buildozer
fi

# Clean previous builds
echo "Step 2: Cleaning previous builds..."
buildozer android clean

# Build APK
echo "Step 3: Building APK (first build takes 20-40 minutes)..."
buildozer -v android debug

# Check result
if [ -f "bin/*.apk" ]; then
    echo "=========================================="
    echo "APK successfully built!"
    echo "File: bin/*.apk"
    echo "=========================================="
    ls -lh bin/*.apk
else
    echo "=========================================="
    echo "APK build failed"
    echo "Check logs above"
    echo "=========================================="
fi
"""
    
    with open('build_in_wsl.sh', 'w', encoding='utf-8', newline='\n') as f:
        f.write(wsl_script)
    
    print(f"   ✅ Создан build_in_wsl.sh")
    
    # 8. Создание Colab ноутбука
    print("\n8️⃣ Создание Colab инструкции...")
    
    colab_instructions = """
# 📱 Google Colab - Инструкция для сборки APK

## Вариант 1: Проект на GitHub

```python
!pip install buildozer cython==0.29.33
!apt-get update && apt-get install -y openjdk-17-jdk

!git clone https://github.com/tairmen/shahed-classifier.git
%cd shahed-classifier

!buildozer -v android debug

from google.colab import files
files.download('bin/sounddetector-1.0.0-arm64-v8a-debug.apk')
```

## Вариант 2: Загрузка ZIP вручную

1. Создайте ZIP архив проекта на вашем ПК:
   - android_sound_detector.py (или main.py)
   - buildozer.spec
   - distance_calibration.json
   - model/my_sound_model.tflite
   - requirements_android.txt

2. В Colab выполните:

```python
from google.colab import files
uploaded = files.upload()  # Загрузите ZIP

!unzip sound_classifier.zip
%cd sound_classifier

!pip install buildozer cython==0.29.33
!apt-get update && apt-get install -y openjdk-17-jdk

!buildozer -v android debug

files.download('bin/sounddetector-1.0.0-arm64-v8a-debug.apk')
```

## Ссылка на Colab:
https://colab.research.google.com/
"""
    
    with open('COLAB_BUILD.md', 'w', encoding='utf-8') as f:
        f.write(colab_instructions)
    
    print(f"   ✅ Создан COLAB_BUILD.md")
    
    # 9. Финальная сводка
    print("\n" + "=" * 60)
    print("✅ ПРОЕКТ ГОТОВ К СБОРКЕ APK!")
    print("=" * 60)
    print("\n📋 Что дальше:\n")
    print("1️⃣ WSL (Windows):")
    print("   - Откройте WSL: wsl")
    print("   - Скопируйте проект в WSL")
    print("   - Запустите: bash build_in_wsl.sh")
    print("\n2️⃣ Google Colab (Проще всего!):")
    print("   - Откройте: https://colab.research.google.com/")
    print("   - Следуйте COLAB_BUILD.md")
    print("\n3️⃣ Linux:")
    print("   - Запустите: buildozer android debug")
    print("\n" + "=" * 60)
    
    return True


def create_zip_for_colab():
    """Создание ZIP архива для загрузки в Colab"""
    import zipfile
    
    print("\n📦 Создание ZIP архива для Colab...")
    
    files_to_zip = [
        'android_sound_detector.py',
        'buildozer.spec',
        'distance_calibration.json',
        'model/my_sound_model.tflite',
        'requirements_android.txt',
        'COLAB_BUILD.md',
    ]
    
    zip_name = 'sound_classifier_android.zip'
    
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in files_to_zip:
            if os.path.exists(file):
                zipf.write(file)
                print(f"   ✅ Добавлен: {file}")
            else:
                print(f"   ⚠️ Пропущен: {file} (не найден)")
    
    size = os.path.getsize(zip_name) / (1024 * 1024)
    print(f"\n✅ Создан архив: {zip_name} ({size:.2f} MB)")
    print(f"   Можете загрузить его в Google Colab!")
    
    return zip_name


if __name__ == '__main__':
    success = prepare_for_android_build()
    
    if success:
        response = input("\nСоздать ZIP архив для Colab? (y/n): ")
        if response.lower() == 'y':
            create_zip_for_colab()
        
        print("\n🎉 Готово! Удачи со сборкой APK!")
