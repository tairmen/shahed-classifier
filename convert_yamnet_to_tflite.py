#!/usr/bin/env python3
"""
Конвертация YAMNet SavedModel → TFLite (embedding extractor)

Запустить ОДИН РАЗ на компьютере (Windows/Linux/Mac) перед копированием на Android.
Требует полный TensorFlow (pip install tensorflow).

Результат: model/yamnet_embeddings.tflite  (~3.7 MB)
"""

import tensorflow as tf
import numpy as np
import os

YAMNET_PATH = './yamnet_local'
OUTPUT_PATH = 'model/yamnet_embeddings.tflite'
SAMPLE_RATE = 16000
WINDOW_SECONDS = 0.975
NUM_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)  # 15600

print(f"🔄 Загрузка YAMNet из {YAMNET_PATH} ...")
yamnet = tf.saved_model.load(YAMNET_PATH)
print("✅ YAMNet загружен")

@tf.function(input_signature=[tf.TensorSpec(shape=[NUM_SAMPLES], dtype=tf.float32)])
def get_embedding(waveform):
    _, embeddings, _ = yamnet(waveform)
    # Усредняем по фреймам → [1024]
    return tf.reduce_mean(embeddings, axis=0)

print("🔄 Конвертация в TFLite...")
concrete_func = get_embedding.get_concrete_function()
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
tflite_model = converter.convert()

os.makedirs('model', exist_ok=True)
with open(OUTPUT_PATH, 'wb') as f:
    f.write(tflite_model)

size_kb = len(tflite_model) / 1024
print(f"✅ Сохранено: {OUTPUT_PATH}  ({size_kb:.0f} KB)")
print(f"   Входной тензор:  [{NUM_SAMPLES}]  float32  (сырая форма волны)")
print(f"   Выходной тензор: [1024]          float32  (embedding)")
print()
print("📋 Теперь скопируйте на телефон:")
print("   model/yamnet_embeddings.tflite")
print("   model/my_sound_model.tflite")
print("   distance_calibration.json")
print("   universal_sound_detector_android.py")
