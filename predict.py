import sys
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

# Пути и модели
MODEL_PATH = "model/my_sound_model.h5"
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
classifier = tf.keras.models.load_model(MODEL_PATH)
class_names = ["negative", "positive"]

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

def extract_embedding(audio_path):
    audio = load_wav_16k_mono(audio_path)
    scores, embeddings, spectrogram = yamnet_model(audio)
    embedding = tf.reduce_mean(embeddings, axis=0)
    return embedding

def main():
    if len(sys.argv) < 2:
        print("Использование: python predict.py путь_к_аудио.wav")
        return
    
    audio_path = sys.argv[1]
    emb = extract_embedding(audio_path)
    pred = classifier.predict(np.expand_dims(emb, axis=0), verbose=0)[0]
    
    for i, class_name in enumerate(class_names):
        print(f"Вероятность '{class_name}': {pred[i]:.3f}")
    
    predicted_class = class_names[np.argmax(pred)]
    confidence = np.max(pred)
    print(f"\nПредсказание: {predicted_class} (уверенность: {confidence:.3f})")

if __name__ == "__main__":
    main()
