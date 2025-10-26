#!/bin/bash
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
