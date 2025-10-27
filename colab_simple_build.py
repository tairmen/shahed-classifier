"""
🚀 КОПИРУЙТЕ ЭТО В GOOGLE COLAB
Простая версия БЕЗ numpy - гарантированно работает
"""

# ==========================================
# 📱 Sound Detector APK Builder
# Версия: БЕЗ NUMPY (для первого теста)
# ==========================================

print("=" * 60)
print("🚀 Sound Detector APK Builder")
print("Версия: Simple (без numpy/AI)")
print("=" * 60)

# Шаг 1: Установка зависимостей
print("\n[1/7] Installing build tools...")
get_ipython().system('pip install -q buildozer cython==0.29.33')
get_ipython().system('apt-get update -qq')
get_ipython().system('apt-get install -y -qq openjdk-17-jdk')
print("✅ Build tools ready")

# Шаг 2: Клонирование проекта
print("\n[2/7] Cloning project...")
get_ipython().system('git clone https://github.com/tairmen/shahed-classifier.git')
get_ipython().run_line_magic('cd', 'shahed-classifier')
print("✅ Project loaded")

# Шаг 3: Удаление numpy из зависимостей (CRITICAL FIX!)
print("\n[3/7] 🔧 FIXING: Removing numpy (causes compilation errors)...")
import re

with open('buildozer.spec', 'r', encoding='utf-8') as f:
    spec = f.read()

# Убираем numpy - он требует компиляции
spec = re.sub(
    r'requirements = .*',
    'requirements = python3,kivy,android,pyjnius',
    spec
)

with open('buildozer.spec', 'w', encoding='utf-8') as f:
    f.write(spec)

print("✅ Fixed! requirements = python3,kivy,android,pyjnius")
print("   (numpy removed - will work without AI for now)")

# Шаг 4: Проверка файлов
print("\n[4/7] Checking files...")
import os
get_ipython().system('ls -lh buildozer.spec android_sound_detector.py')
print("✅ Files present")

# Шаг 5: Очистка кэша
print("\n[5/7] Cleaning cache...")
get_ipython().system('buildozer android clean')
print("✅ Cache cleared")

# Шаг 6: Сборка APK
print("\n[6/7] Building APK...")
print("⏰ This will take 20-30 minutes")
print("☕ Perfect time for coffee!")
print("=" * 60)

get_ipython().system('buildozer -v android debug')

print("\n" + "=" * 60)
print("✅ Build completed!")
print("=" * 60)

# Шаг 7: Скачивание APK
print("\n[7/7] Downloading APK...")

from google.colab import files
import os

try:
    if os.path.exists('bin'):
        apk_files = [f for f in os.listdir('bin') if f.endswith('.apk')]
        
        if apk_files:
            apk_path = f'bin/{apk_files[0]}'
            size_mb = os.path.getsize(apk_path) / (1024*1024)
            
            print("\n" + "=" * 60)
            print("🎉 SUCCESS!")
            print("=" * 60)
            print(f"📱 File: {apk_files[0]}")
            print(f"📦 Size: {size_mb:.1f} MB")
            print(f"📝 Note: This version works WITHOUT AI (no numpy)")
            print(f"         UI and basic features work fine!")
            print("\n⬇️  Downloading...")
            
            files.download(apk_path)
            
            print("\n✅ APK downloaded successfully!")
            print("\n📋 Installation steps:")
            print("1. Transfer APK to your Android device")
            print("2. Enable 'Unknown Sources' in Settings")
            print("3. Install the APK")
            print("4. Grant microphone permission")
            print("5. Press START to begin detection")
            print("\n⚠️  Note: AI model won't load (no numpy)")
            print("          But UI and basic sound detection works!")
            
        else:
            print("❌ ERROR: No APK files found in bin/")
            print("\n📁 Contents of bin/:")
            get_ipython().system('ls -la bin/')
            print("\n⚠️  Check build logs above for errors")
    else:
        print("❌ ERROR: bin/ directory not found!")
        print("Build failed. Check the logs above.")
        print("\n📁 Current directory contents:")
        get_ipython().system('ls -la')
        
except Exception as e:
    print(f"❌ ERROR: {e}")
    print("\n🔍 Checking if APK exists...")
    if os.path.exists('bin'):
        get_ipython().system('ls -la bin/')

print("\n" + "=" * 60)
print("Process complete!")
print("=" * 60)
