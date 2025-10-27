"""
🔧 Подготовка файлов для Android APK
Убеждаемся что модель включена и путь правильный
"""

import os
import shutil

print("=" * 60)
print("📱 Подготовка файлов для Android")
print("=" * 60)

# 1. Проверка модели
print("\n1️⃣ Проверка модели...")

model_files = {
    'model/my_sound_model.h5': 'H5 модель',
    'model/my_sound_model.tflite': 'TFLite модель',
}

models_found = []
for path, desc in model_files.items():
    if os.path.exists(path):
        size = os.path.getsize(path) / (1024 * 1024)
        print(f"   ✅ {desc}: {path} ({size:.2f} MB)")
        models_found.append(path)
    else:
        print(f"   ❌ {desc}: {path} - НЕ НАЙДЕНА!")

if not models_found:
    print("\n⚠️ КРИТИЧЕСКАЯ ОШИБКА: Модели не найдены!")
    print("Запустите: python convert_to_tflite.py")
    exit(1)

# 2. Проверка buildozer.spec
print("\n2️⃣ Проверка buildozer.spec...")

with open('buildozer.spec', 'r', encoding='utf-8') as f:
    spec = f.read()

if 'model/*.tflite' in spec or 'model/*.h5' in spec:
    print("   ✅ Модели включены в buildozer.spec")
else:
    print("   ⚠️ Модели НЕ включены в buildozer.spec!")
    print("   Добавляю...")
    
    # Добавляем модели
    spec = spec.replace(
        'source.include_patterns = ',
        'source.include_patterns = model/*.tflite,model/*.h5,'
    )
    
    with open('buildozer.spec', 'w', encoding='utf-8') as f:
        f.write(spec)
    
    print("   ✅ Модели добавлены в buildozer.spec")

# 3. Создаем main.py если нужно
print("\n3️⃣ Проверка главного файла...")

if os.path.exists('main.py'):
    print("   ℹ️ main.py уже существует")
else:
    if os.path.exists('android_sound_detector.py'):
        shutil.copy('android_sound_detector.py', 'main.py')
        print("   ✅ Создан main.py")
    else:
        print("   ❌ android_sound_detector.py не найден!")

# 4. Проверка других файлов
print("\n4️⃣ Проверка дополнительных файлов...")

required = [
    'distance_calibration.json',
    'buildozer.spec',
]

for file in required:
    if os.path.exists(file):
        print(f"   ✅ {file}")
    else:
        print(f"   ❌ {file} - отсутствует!")

# 5. Создаем тестовую структуру файлов
print("\n5️⃣ Создание тестового скрипта...")

test_script = """
# Тест загрузки модели на Android
import os

print("Текущая директория:", os.getcwd())
print("\\nСодержимое директории:")
for root, dirs, files in os.walk('.'):
    level = root.replace('.', '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    for file in files:
        if file.endswith(('.h5', '.tflite', '.json')):
            print(f'{subindent}{file}')
"""

with open('test_model_paths.py', 'w', encoding='utf-8') as f:
    f.write(test_script)

print("   ✅ Создан test_model_paths.py")

# Финальная сводка
print("\n" + "=" * 60)
print("✅ ПОДГОТОВКА ЗАВЕРШЕНА!")
print("=" * 60)
print("\n📋 Следующие шаги:")
print("1. Закоммитьте изменения:")
print("   git add .")
print('   git commit -m "Fix: Include model in APK"')
print("   git push")
print("\n2. Пересоберите APK в Colab:")
print("   - Используйте обновленный код")
print("   - Модели теперь включены")
print("\n3. После установки на Android:")
print("   - Проверьте логи: adb logcat | grep python")
print("   - Внизу экрана будет показана директория")
print("\n" + "=" * 60)
