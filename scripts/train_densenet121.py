import os
import time
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# === Set seed for reproducibility ===
tf.random.set_seed(42)

# === Create output folders ===
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# === Dataset paths ===
train_dir = 'DataSplit/train'
val_dir = 'DataSplit/val'

# === Image data generators ===
datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_data = datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# === Load DenseNet121 base ===
base_model = DenseNet121(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# ✅ Fine-tune top layers only
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

# === Build the model ===
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(3, activation='softmax')
])

# === Compile with small learning rate
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# === Early stopping ===
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=4,
    restore_best_weights=True
)

# === Train ===
start_time = time.time()

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=25,
    callbacks=[early_stop]
)

end_time = time.time()
training_duration = end_time - start_time
print(f"⏱️ Training completed in {training_duration:.2f} seconds")

# === Save the model ===
model.save('models/densenet121_mask_model.h5')
print("✅ Model saved to models/densenet121_mask_model.h5")

# === Plot training curves ===
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('DenseNet121 - Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('DenseNet121 - Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('results/densenet121_training_curves.png')
plt.show()

