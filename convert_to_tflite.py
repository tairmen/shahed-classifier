"""
🔧 Конвертация модели H5 в TensorFlow Lite для Android
Запустите этот скрипт перед сборкой APK
"""

import tensorflow as tf
import os
import numpy as np

def convert_model_to_tflite():
    """Конвертация H5 модели в TFLite формат"""
    
    model_path = 'model/my_sound_model.h5'
    tflite_path = 'model/my_sound_model.tflite'
    
    print("🔄 Конвертация модели для Android...")
    print(f"📂 Исходная модель: {model_path}")
    
    # Проверка существования модели
    if not os.path.exists(model_path):
        print(f"❌ Модель не найдена: {model_path}")
        print("Сначала обучите модель с помощью train_balanced.py")
        return False
    
    try:
        # Загрузка H5 модели
        print("📥 Загрузка H5 модели...")
        model = tf.keras.models.load_model(model_path, compile=False)
        
        print(f"✅ Модель загружена")
        print(f"📊 Архитектура:")
        model.summary()
        
        # Конвертация в TFLite
        print("\n🔧 Конвертация в TFLite...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Оптимизация для мобильных устройств
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Квантизация для уменьшения размера (опционально)
        # converter.target_spec.supported_types = [tf.float16]
        
        # Конвертируем
        tflite_model = converter.convert()
        
        # Сохраняем
        print(f"💾 Сохранение в {tflite_path}...")
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        # Информация о размерах
        h5_size = os.path.getsize(model_path) / (1024 * 1024)
        tflite_size = os.path.getsize(tflite_path) / (1024 * 1024)
        
        print(f"\n✅ Конвертация завершена!")
        print(f"📦 Размер H5: {h5_size:.2f} MB")
        print(f"📦 Размер TFLite: {tflite_size:.2f} MB")
        print(f"📉 Уменьшение: {(1 - tflite_size/h5_size)*100:.1f}%")
        
        # Тест модели
        print("\n🧪 Тестирование TFLite модели...")
        test_tflite_model(tflite_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка конвертации: {e}")
        return False


def test_tflite_model(tflite_path):
    """Тестирование TFLite модели"""
    try:
        # Загрузка интерпретатора
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # Информация о входе/выходе
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"📥 Вход: shape={input_details[0]['shape']}, dtype={input_details[0]['dtype']}")
        print(f"📤 Выход: shape={output_details[0]['shape']}, dtype={output_details[0]['dtype']}")
        
        # Тестовый вход (1024 features от YAMNet)
        test_input = np.random.randn(1, 1024).astype(np.float32)
        
        # Запуск инференса
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        print(f"✅ Тест пройден!")
        print(f"📊 Тестовый выход: {output[0]}")
        print(f"🎯 Класс: {'Positive' if output[0][1] > 0.5 else 'Negative'} (уверенность: {max(output[0]):.2%})")
        
    except Exception as e:
        print(f"⚠️ Ошибка теста: {e}")


def create_quantized_model():
    """Создание квантизованной модели (еще меньше размер)"""
    
    model_path = 'model/my_sound_model.h5'
    quantized_path = 'model/my_sound_model_quantized.tflite'
    
    print("\n🔧 Создание квантизованной модели...")
    
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        
        # Конвертация с квантизацией
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]  # Float16 квантизация
        
        quantized_model = converter.convert()
        
        with open(quantized_path, 'wb') as f:
            f.write(quantized_model)
        
        size = os.path.getsize(quantized_path) / (1024 * 1024)
        print(f"✅ Квантизованная модель сохранена: {quantized_path}")
        print(f"📦 Размер: {size:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка квантизации: {e}")
        return False


if __name__ == '__main__':
    print("=" * 60)
    print("🤖 TensorFlow Lite Converter для Android")
    print("=" * 60)
    
    # Конвертация основной модели
    success = convert_model_to_tflite()
    
    if success:
        print("\n" + "=" * 60)
        
        # Опционально: создание квантизованной версии
        response = input("Создать квантизованную версию? (y/n): ")
        if response.lower() == 'y':
            create_quantized_model()
        
        print("\n" + "=" * 60)
        print("✅ Готово! Теперь можете собрать APK:")
        print("   buildozer android debug")
        print("=" * 60)
    else:
        print("\n❌ Конвертация не удалась")
