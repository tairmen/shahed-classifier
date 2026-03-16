"""
🚀 БЫСТРЫЙ СТАРТ - Google Colab сборка APK
Скопируйте этот код в новый Colab notebook
"""

# ============================================
# 📱 Сборка Android APK - ИСПРАВЛЕННАЯ ВЕРСИЯ
# ============================================

print("=" * 60)
print("🎯 Android Sound Detector - APK Builder")
print("=" * 60)

# Шаг 1: Установка Buildozer
print("\n[1/6] Установка buildozer...")
get_ipython().system('pip install -q buildozer cython==0.29.33')

# Шаг 2: Установка Android зависимостей
print("\n[2/6] Установка Android SDK/NDK зависимостей...")
get_ipython().system('apt-get update -qq')
get_ipython().system('apt-get install -y -qq openjdk-17-jdk autoconf libtool')

# Шаг 3: Загрузка проекта
print("\n[3/6] Загрузка проекта...")
print("Выберите метод:")
print("  A) Из GitHub (автоматически)")
print("  B) Загрузить ZIP вручную")

# ВАРИАНТ A: Из GitHub
get_ipython().system('git clone https://github.com/tairmen/shahed-classifier.git')
get_ipython().run_line_magic('cd', 'shahed-classifier')

# ВАРИАНТ B: Ручная загрузка (раскомментируйте если нужно)
# from google.colab import files
# print("Загрузите sound_classifier_android_fixed.zip:")
# uploaded = files.upload()
# !unzip -q sound_classifier_android_fixed.zip
# %cd sound_classifier

# Шаг 4: Проверка файлов
print("\n[4/6] Проверка файлов...")
import os
required = ['android_sound_detector.py', 'buildozer.spec', 'model/my_sound_model.tflite']
for f in required:
    if os.path.exists(f):
        print(f"  ✅ {f}")
    else:
        print(f"  ❌ {f} - НЕ НАЙДЕН!")

# Шаг 5: Сборка APK
print("\n[5/6] Сборка APK (это займет 20-30 минут)...")
print("=" * 60)
print("☕ Время выпить кофе!")
print("=" * 60)
get_ipython().system('buildozer -v android debug')

# Шаг 6: Скачивание APK
print("\n[6/6] Проверка и скачивание APK...")
print("=" * 60)

from google.colab import files
import os

if os.path.exists('bin'):
    apk_files = [f for f in os.listdir('bin/') if f.endswith('.apk')]
    
    if apk_files:
        apk_path = f'bin/{apk_files[0]}'
        size_mb = os.path.getsize(apk_path) / (1024 * 1024)
        
        print(f"✅ APK УСПЕШНО СОБРАН!")
        print(f"📦 Файл: {apk_files[0]}")
        print(f"📊 Размер: {size_mb:.2f} MB")
        print(f"📱 Минимальная версия Android: 5.0 (API 21)")
        print(f"🎯 Целевая версия Android: 13 (API 33)")
        print("\n⬇️ Начинаем скачивание...")
        print("=" * 60)
        
        files.download(apk_path)
        
        print("\n" + "=" * 60)
        print("🎉 ГОТОВО!")
        print("=" * 60)
        print("\n📝 Следующие шаги:")
        print("1. Скопируйте APK на Android телефон")
        print("2. Включите 'Неизвестные источники' в настройках")
        print("3. Установите APK")
        print("4. Дайте разрешение на микрофон при первом запуске")
        print("\n✨ Готово к использованию!")
    else:
        print("❌ APK не найден в bin/")
        print("\n🔍 Содержимое bin/:")
        get_ipython().system('ls -la bin/')
        print("\n⚠️ Проверьте логи сборки выше на наличие ошибок")
else:
    print("❌ Папка bin/ не существует!")
    print("Сборка не удалась. Проверьте логи выше.")
    print("\n🔍 Текущие файлы:")
    get_ipython().system('ls -la')

print("=" * 60)
