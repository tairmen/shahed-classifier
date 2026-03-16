
# Тест загрузки модели на Android
import os

print("Текущая директория:", os.getcwd())
print("\nСодержимое директории:")
for root, dirs, files in os.walk('.'):
    level = root.replace('.', '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    for file in files:
        if file.endswith(('.h5', '.tflite', '.json')):
            print(f'{subindent}{file}')
