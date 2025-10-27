[app]

# (str) Название приложения
title = Sound Detector

# (str) Название пакета
package.name = sounddetector

# (str) Домен пакета (уникальный идентификатор)
package.domain = org.soundai

# (str) Исходный код приложения (где находится main.py)
source.dir = .

# (list) Исходные файлы для включения (шаблон)
source.include_exts = py,png,jpg,kv,atlas,json,h5,wav,txt

# (str) Версия приложения
version = 1.0.0

# (list) Зависимости приложения
# Минимальные зависимости для избежания 404 ошибок
# Используем последние стабильные версии
requirements = python3,kivy,numpy,android,pyjnius

# (list) Разрешения приложения
android.permissions = RECORD_AUDIO,WRITE_EXTERNAL_STORAGE,READ_EXTERNAL_STORAGE,INTERNET

# (int) Целевая версия Android API
android.api = 33

# (int) Минимальная версия Android API
android.minapi = 21

# (str) Архитектура Android (может быть armeabi-v7a, arm64-v8a, x86, x86_64)
android.archs = arm64-v8a,armeabi-v7a

# (bool) Включить логирование
android.logcat_filters = *:S python:D

# (str) Путь к иконке приложения
#icon.filename = %(source.dir)s/data/icon.png

# (str) Путь к splash screen
#presplash.filename = %(source.dir)s/data/presplash.png

# (str) Ориентация экрана (portrait, landscape, sensor)
orientation = portrait

# (bool) Разрешить полноэкранный режим
fullscreen = 0

# (list) Включить дополнительные файлы/папки
# Формат: source:destination
source.include_patterns = model/*.tflite,model/*.h5,distance_calibration.json

# (str) Главный файл приложения
# Buildozer автоматически переименует android_sound_detector.py в main.py при сборке
android.entrypoint = org.kivy.android.PythonActivity

# (str) Поддерживаемые ориентации
android.manifest.orientation = portrait

# (bool) Пропускать обновление buildozer
android.skip_update = False

# (bool) Использовать AndroidX
android.gradle_dependencies = androidx.appcompat:appcompat:1.4.1,androidx.core:core:1.7.0

# (str) Bootstrap (sdl2 работает лучше для аудио)
p4a.bootstrap = sdl2

# (list) Рецепты python-for-android для включения
# Добавляем numpy и другие необходимые библиотеки
p4a.local_recipes = 

# (bool) Копировать библиотеки numpy
android.copy_libs = 1

# (str) Поддерживаемые языки
android.manifest.intent_filters = 

# (list) Дополнительные опции gradle
# android.gradle_dependencies = 

[buildozer]

# (int) Уровень логирования (0 = только ошибки, 1 = info, 2 = debug)
log_level = 2

# (int) Отображать предупреждения
warn_on_root = 1

# (str) Путь к buildozer
# build_dir = ./.buildozer

# (str) Путь к bin
# bin_dir = ./bin
