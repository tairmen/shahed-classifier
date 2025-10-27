"""
🚀 Быстрый фикс для Google Colab - 404 ошибка
Скопируйте этот код целиком в Colab
"""

# ============================================
# 📱 APK Builder - FINAL FIXED VERSION
# ============================================

print("🚀 APK Build Process Starting...")
print("=" * 60)
print("⚠️  FIX: Removing version pins to avoid 404 errors")
print("=" * 60)

# 1. Install tools
print("\n[1/6] Installing dependencies...")
get_ipython().system('pip install -q buildozer cython==0.29.33')
get_ipython().system('apt-get update -qq')
get_ipython().system('apt-get install -y -qq openjdk-17-jdk autoconf libtool')
print("✅ Done")

# 2. Clone project
print("\n[2/6] Loading project...")
get_ipython().system('git clone https://github.com/tairmen/shahed-classifier.git')
get_ipython().run_line_magic('cd', 'shahed-classifier')
print("✅ Done")

# 3. CRITICAL FIX - Update buildozer.spec
print("\n[3/6] 🔧 FIXING buildozer.spec (404 fix)...")
import re

with open('buildozer.spec', 'r') as f:
    spec = f.read()

# Remove ALL version pins - use latest stable
spec = re.sub(
    r'requirements = .*',
    'requirements = python3,kivy,numpy,android,pyjnius',
    spec
)

with open('buildozer.spec', 'w') as f:
    f.write(spec)

print("✅ Fixed! Using: python3,kivy,numpy,android,pyjnius")
print("   (No version pins = no 404 errors)")

# 4. Clean
print("\n[4/6] Cleaning cache...")
get_ipython().system('buildozer android clean')
print("✅ Done")

# 5. Build
print("\n[5/6] Building APK...")
print("⏰ This takes 20-30 minutes")
print("☕ Perfect time for coffee!")
print("=" * 60)

get_ipython().system('buildozer -v android debug')

print("\n" + "=" * 60)
print("✅ Build complete!")
print("=" * 60)

# 6. Download
print("\n[6/6] Downloading APK...")
from google.colab import files
import os

if os.path.exists('bin'):
    apks = [f for f in os.listdir('bin') if f.endswith('.apk')]
    if apks:
        apk = f'bin/{apks[0]}'
        size = os.path.getsize(apk) / (1024*1024)
        print(f"\n🎉 SUCCESS!")
        print(f"📱 File: {apks[0]}")
        print(f"📦 Size: {size:.1f} MB")
        print(f"⬇️  Downloading...")
        files.download(apk)
        print(f"\n✨ APK downloaded! Install on Android.")
    else:
        print("❌ No APK. Check logs above.")
else:
    print("❌ Build failed. Check logs above.")

print("=" * 60)
print("🎉 Process complete!")
print("=" * 60)
