import tensorflow as tf

# Загружаем TFLite модель
interpreter = tf.lite.Interpreter(model_path='model/my_sound_model.tflite')
interpreter.allocate_tensors()

# Получаем информацию о входе/выходе
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("=" * 60)
print("📥 INPUT (что модель принимает):")
print("=" * 60)
print(f"  Shape: {input_details[0]['shape']}")
print(f"  Type: {input_details[0]['dtype']}")
print(f"  Name: {input_details[0]['name']}")

print("\n" + "=" * 60)
print("📤 OUTPUT (что модель возвращает):")
print("=" * 60)
print(f"  Shape: {output_details[0]['shape']}")
print(f"  Type: {output_details[0]['dtype']}")
print(f"  Name: {output_details[0]['name']}")

print("\n" + "=" * 60)
print("🔍 АНАЛИЗ СООТВЕТСТВИЯ ТРЕБОВАНИЯМ:")
print("=" * 60)

# Проверяем требования
requirements = {
    "Вход - FloatArray": input_details[0]['dtype'] == tf.float32,
    "Размер входа - 1024": input_details[0]['shape'][1] == 1024,
    "Выход - массив вероятностей": output_details[0]['dtype'] == tf.float32,
    "Выход - 2 класса (positive/negative)": output_details[0]['shape'][1] == 2
}

for req, status in requirements.items():
    icon = "✅" if status else "❌"
    print(f"{icon} {req}: {status}")

print("\n" + "=" * 60)
print("⚠️ ВАЖНОЕ НЕСООТВЕТСТВИЕ:")
print("=" * 60)
print("Требование: Вход должен быть RAW AUDIO (16000 samples)")
print(f"Ваша модель: Вход - YAMNet embeddings (1024 features)")
print("\n🔧 Ваша архитектура:")
print("   Audio (16000) → YAMNet → Embeddings (1024) → Ваша модель → Prediction")
print("\n📱 Для Android нужно:")
print("   1. Включить YAMNet в APK")
print("   2. Сделать pre-processing на устройстве")
print("   3. Или переобучить модель end-to-end на raw audio")
