import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
from tensorflow.keras import layers, models
from sklearn.utils.class_weight import compute_class_weight

DATA_DIR = "data"
MODEL_PATH = "model/my_sound_model.h5"

yamnet_model_handle = "https://tfhub.dev/google/yamnet/1"
yamnet_model = hub.load(yamnet_model_handle)

def load_wav_16k_mono(filename):
    file_contents = tf.io.read_file(filename)
    audio, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    audio = tf.squeeze(audio, axis=-1)
    
    # Convert to 16kHz if needed
    sample_rate = tf.cast(sample_rate, tf.float32)
    if tf.not_equal(sample_rate, 16000.0):
        # Calculate target length for 16kHz
        current_length = tf.cast(tf.shape(audio)[0], tf.float32)
        target_length = tf.cast(current_length * 16000.0 / sample_rate, tf.int32)
        
        # Resample using signal processing
        indices = tf.cast(tf.linspace(0.0, current_length - 1, target_length), tf.int32)
        audio = tf.gather(audio, indices)
    
    return audio

def extract_embedding(filename):
    audio = load_wav_16k_mono(filename)
    scores, embeddings, spectrogram = yamnet_model(audio)
    embedding = tf.reduce_mean(embeddings, axis=0)
    return embedding

def augment_audio_embedding(embedding, num_augmentations=5):
    """Создает дополнительные вариации эмбеддинга для аугментации данных"""
    augmented = [embedding]
    
    for _ in range(num_augmentations):
        # Добавляем небольшой шум
        noise = tf.random.normal(shape=embedding.shape, mean=0.0, stddev=0.001)
        augmented_emb = embedding + noise
        augmented.append(augmented_emb)
    
    return augmented

# Загружаем данные
X, y = [], []
class_names = sorted(os.listdir(DATA_DIR))

# Сначала загружаем все данные
all_embeddings = {class_name: [] for class_name in class_names}

for label_idx, class_name in enumerate(class_names):
    folder = os.path.join(DATA_DIR, class_name)
    print(f"📂 Загружаем {class_name}...")
    
    for file in os.listdir(folder):
        if file.endswith(".wav"):
            path = os.path.join(folder, file)
            emb = extract_embedding(path)
            all_embeddings[class_name].append(emb.numpy())

# Подсчитываем статистику
neg_count = len(all_embeddings['negative'])
pos_count = len(all_embeddings['positive'])

print(f"📊 Исходные данные:")
print(f"  negative: {neg_count} файлов")
print(f"  positive: {pos_count} файлов")

# Аугментируем positive класс для баланса
target_count = min(neg_count, 500)  # Ограничиваем до разумного размера
augmentation_factor = target_count // pos_count

print(f"🔄 Аугментация positive класс: {augmentation_factor}x")

# Добавляем negative примеры (берем случайную выборку если их слишком много)
negative_embeddings = all_embeddings['negative']
if len(negative_embeddings) > target_count:
    indices = np.random.choice(len(negative_embeddings), target_count, replace=False)
    negative_embeddings = [negative_embeddings[i] for i in indices]

for emb in negative_embeddings:
    X.append(emb)
    y.append(0)

# Добавляем positive примеры с аугментацией
for emb in all_embeddings['positive']:
    emb_tensor = tf.constant(emb)
    augmented_embs = augment_audio_embedding(emb_tensor, augmentation_factor - 1)
    
    for aug_emb in augmented_embs:
        X.append(aug_emb.numpy())
        y.append(1)

X = np.array(X)
y = np.array(y)

# Перемешиваем данные
indices = np.random.permutation(len(X))
X = X[indices]
y = y[indices]

print(f"📊 Финальное распределение:")
print(f"  negative: {np.sum(y == 0)} файлов")
print(f"  positive: {np.sum(y == 1)} файлов")

# Создаем модель
model = models.Sequential([
    layers.Input(shape=(1024,)),
    layers.Dense(512, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(256, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(len(class_names), activation="softmax")
])

# Компилируем с меньшим learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
    loss="sparse_categorical_crossentropy", 
    metrics=["accuracy"]
)

# Обучаем модель
history = model.fit(
    X, y, 
    epochs=100, 
    batch_size=32, 
    validation_split=0.2,
    verbose=1
)

# Сохраняем модель
os.makedirs("model", exist_ok=True)
model.save(MODEL_PATH)

print(f"✅ Сбалансированная модель сохранена в {MODEL_PATH}")