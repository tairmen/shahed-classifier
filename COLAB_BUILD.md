
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
