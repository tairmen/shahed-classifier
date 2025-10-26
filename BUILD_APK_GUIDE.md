# 🚀 Сборка APK на Windows - Пошаговая инструкция

## ⚠️ Важно: Buildozer работает только на Linux!

Так как вы на Windows, у вас есть **3 варианта**:

---

## 🎯 ВАРИАНТ 1: WSL2 (Windows Subsystem for Linux) - РЕКОМЕНДУЕТСЯ

### Шаг 1: Установите WSL2
```powershell
# В PowerShell от администратора:
wsl --install -d Ubuntu
```

После установки перезагрузите компьютер.

### Шаг 2: Запустите Ubuntu WSL
```powershell
wsl
```

### Шаг 3: В WSL установите зависимости
```bash
# Обновление системы
sudo apt update && sudo apt upgrade -y

# Установка необходимых пакетов
sudo apt install -y python3 python3-pip build-essential git zip unzip \
    openjdk-17-jdk autoconf libtool pkg-config zlib1g-dev \
    libncurses5-dev libncursesw5-dev libtinfo5 cmake libffi-dev \
    libssl-dev libltdl-dev

# Установка Cython и Buildozer
pip3 install --upgrade pip
pip3 install --upgrade cython==0.29.33
pip3 install --upgrade buildozer
```

### Шаг 4: Скопируйте проект в WSL
```bash
# Создайте папку в WSL
mkdir -p ~/sound_classifier
cd ~/sound_classifier

# Скопируйте файлы из Windows (замените путь на ваш)
cp -r /mnt/c/Users/abdur/Projects/sound_classifier/* .
```

### Шаг 5: Соберите APK
```bash
# Очистка (если были предыдущие попытки)
buildozer android clean

# Сборка APK (первая сборка займет 30-60 минут!)
buildozer -v android debug
```

### Шаг 6: Забрать APK обратно в Windows
```bash
# APK будет в bin/
cp bin/*.apk /mnt/c/Users/abdur/Projects/sound_classifier/
```

---

## 🌐 ВАРИАНТ 2: Google Colab (Облачная сборка) - САМЫЙ ПРОСТОЙ

### Готовый скрипт для Colab:

1. Откройте: https://colab.research.google.com
2. Создайте новый Notebook
3. Вставьте и выполните:

```python
# ============================================
# 📱 Сборка Android APK в Google Colab
# ============================================

# 1. Установка Buildozer
!pip install buildozer
!pip install cython==0.29.33

# 2. Установка зависимостей Android
!apt-get update
!apt-get install -y openjdk-17-jdk

# 3. Загрузите ваш проект
# Вариант A: Если проект в GitHub
!git clone https://github.com/tairmen/shahed-classifier.git
%cd shahed-classifier

# Вариант B: Если загружаете вручную
# from google.colab import files
# uploaded = files.upload()  # Загрузите ZIP архив проекта
# !unzip sound_classifier.zip
# %cd sound_classifier

# 4. Сборка APK
!buildozer -v android debug

# 5. Скачать готовый APK
from google.colab import files
import os

apk_files = [f for f in os.listdir('bin/') if f.endswith('.apk')]
if apk_files:
    files.download(f'bin/{apk_files[0]}')
    print(f"✅ APK готов к скачиванию: {apk_files[0]}")
else:
    print("❌ APK не найден. Проверьте логи сборки выше.")
```

---

## 🐳 ВАРИАНТ 3: Docker контейнер

### Создайте файл `build_apk.bat` в вашей папке:

```batch
@echo off
echo ========================================
echo Building APK using Docker
echo ========================================

docker run --rm -v "%cd%:/app" -w /app kivy/buildozer android debug

echo ========================================
echo Done! Check bin/ folder for APK
echo ========================================
pause
```

### Запустите:
```powershell
# Установите Docker Desktop (если еще нет)
# https://www.docker.com/products/docker-desktop/

# Соберите APK
.\build_apk.bat
```

---

## 📦 Упрощенный вариант: Архив для сборки

Если хотите, я могу создать ZIP архив с инструкциями для передачи на Linux машину или облако.

---

## ⚡ Какой вариант выбрать?

| Вариант | Скорость | Сложность | Рекомендация |
|---------|----------|-----------|--------------|
| **WSL2** | ⭐⭐⭐ | ⭐⭐ | Лучший для разработки |
| **Google Colab** | ⭐⭐⭐⭐⭐ | ⭐ | Самый простой! |
| **Docker** | ⭐⭐⭐⭐ | ⭐⭐⭐ | Если есть опыт |

---

## 🎯 Мой совет: **Google Colab** (5 минут)

1. Загрузите проект на GitHub
2. Откройте Colab
3. Выполните скрипт выше
4. Скачайте APK

**Время сборки:** 20-40 минут (зависит от интернета)
**Требования:** Только браузер!

---

## 📝 Что делать после сборки?

После получения APK:

```powershell
# Установите на Android через ADB
adb install sounddetector-1.0.0-arm64-v8a-debug.apk

# Или скопируйте на телефон и установите вручную
# (Включите "Неизвестные источники" в настройках)
```

---

Какой вариант попробуем? Я рекомендую **Google Colab** - это проще всего! 🚀
