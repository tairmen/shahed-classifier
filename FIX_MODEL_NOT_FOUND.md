# 🔧 ИСПРАВЛЕНИЕ: Модель не загружается в Android APK

## ❌ Проблема
После установки APK на телефон приложение показывает "AI модель не найдена"

## 🔍 Причина
1. **Модель не включена в APK** - `buildozer.spec` указывал только `.h5`, но не `.tflite`
2. **Неправильные пути** - на Android файлы находятся в другой директории
3. **TensorFlow не установлен** - нужна поддержка TFLite

## ✅ Что исправлено

### 1. buildozer.spec
```ini
# БЫЛО:
source.include_patterns = model/*.h5,distance_calibration.json,data/positive/*.wav

# СТАЛО:
source.include_patterns = model/*.tflite,model/*.h5,distance_calibration.json
```

### 2. android_sound_detector.py
- ✅ Добавлена поддержка нескольких путей к модели
- ✅ Приоритет TFLite модели (меньше размер)
- ✅ Fallback на H5 для тестирования
- ✅ Отображение текущей директории в UI для отладки

### 3. Загрузка модели
Теперь проверяются все возможные пути:
```python
possible_paths = [
    'model/my_sound_model.tflite',  # Относительный путь
    'model/my_sound_model.h5',       
    '/data/data/org.soundai.sounddetector/files/app/model/my_sound_model.tflite',  # Android
    '/data/data/org.soundai.sounddetector/files/app/model/my_sound_model.h5',
]
```

---

## 🚀 Как пересобрать APK с исправлениями

### Шаг 1: Обновите код на GitHub
```bash
git add .
git commit -m "Fix: Include model in APK and improve path resolution"
git push
```

### Шаг 2: Пересоберите в Colab

Используйте этот код в Google Colab:

```python
# Установка
!pip install -q buildozer cython==0.29.33
!apt-get update -qq && apt-get install -y -qq openjdk-17-jdk

# Клонирование (получит исправленный код)
!git clone https://github.com/tairmen/shahed-classifier.git
%cd shahed-classifier

# Убедимся что модель включена
!ls -lh model/*.tflite model/*.h5

# Проверка buildozer.spec
!grep "source.include_patterns" buildozer.spec

# Сборка
!buildozer -v android debug

# Скачивание
from google.colab import files
import os
apk = [f for f in os.listdir('bin/') if f.endswith('.apk')][0]
files.download(f'bin/{apk}')
```

### Шаг 3: Установите новый APK

1. Удалите старую версию приложения с телефона
2. Установите новый APK
3. Проверьте внизу экрана:
   - Должно быть: **"AI модель: ✅ Загружена"**
   - Показывается текущая директория

---

## 🧪 Проверка перед сборкой

Запустите на ПК:
```bash
python prepare_model_for_android.py
```

Должно показать:
```
✅ H5 модель: model/my_sound_model.h5 (7.97 MB)
✅ TFLite модель: model/my_sound_model.tflite (0.67 MB)
✅ Модели включены в buildozer.spec
```

---

## 🐛 Если модель все еще не загружается

### Вариант 1: Проверьте логи Android
```bash
adb logcat | grep python
```

Ищите строки:
- "✅ Найдена модель: ..."
- "⚠️ Модель не найдена"
- Список проверенных путей

### Вариант 2: Проверьте содержимое APK
```bash
# Распакуйте APK
unzip sounddetector-*.apk -d apk_contents

# Проверьте наличие модели
ls -lh apk_contents/assets/model/
```

Должны быть файлы:
- `my_sound_model.tflite` (0.67 MB)
- `my_sound_model.h5` (7.97 MB) - опционально

### Вариант 3: Минимальная отладочная версия

Если ничего не помогает, создайте версию без AI:

```python
# В android_sound_detector.py измените:
TENSORFLOW_AVAILABLE = False  # Принудительно отключить AI

# Или закомментируйте строку:
# import tensorflow as tf
```

Это создаст рабочее приложение с UI, но без классификации (демо режим).

---

## 📱 Что должно работать после исправления

✅ Модель загружается при старте  
✅ Внизу показано "AI модель: ✅ Загружена"  
✅ Отображается текущая директория  
✅ Детекция работает с AI классификацией  
✅ Размер APK ~65-90 MB (с моделью)  

---

## 💡 Дополнительные улучшения (опционально)

### Уменьшить размер APK
Используйте только TFLite модель:
```ini
# В buildozer.spec:
source.include_patterns = model/*.tflite,distance_calibration.json
```

Размер уменьшится на ~7 MB.

### Добавить проверку при старте
В `on_start()` метод приложения добавьте:
```python
def on_start(self):
    if not self.model_loaded:
        from kivy.uix.popup import Popup
        popup = Popup(title='Предупреждение',
                      content=Label(text='AI модель не найдена!\nРаботает в демо режиме.'),
                      size_hint=(0.8, 0.3))
        popup.open()
```

---

## ✅ Checklist перед сборкой

- [ ] `model/my_sound_model.tflite` существует (0.67 MB)
- [ ] `buildozer.spec` содержит `model/*.tflite`
- [ ] `android_sound_detector.py` обновлен
- [ ] Изменения закоммичены в GitHub
- [ ] Используется свежий клон в Colab

**После выполнения всех шагов модель должна загружаться!** 🎉
