import zipfile
import os

# Файлы для включения в архив
files_to_include = [
    'android_sound_detector.py',
    'buildozer.spec',
    'distance_calibration.json',
    'model/my_sound_model.tflite',
    'COLAB_BUILD_FIXED.md'
]

# Создание ZIP
zip_name = 'sound_classifier_android_fixed.zip'

with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for file in files_to_include:
        if os.path.exists(file):
            zipf.write(file)
            print(f"Added: {file}")
        else:
            print(f"Skipped: {file} (not found)")

size = os.path.getsize(zip_name) / (1024 * 1024)
print(f"\nCreated: {zip_name} ({size:.2f} MB)")
print("Ready for Google Colab upload!")
