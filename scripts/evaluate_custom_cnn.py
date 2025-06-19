import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# === Load your model ===
model_path = 'models/cnn_baseline_mask_model.h5'
model = load_model(model_path)
print(f"✅ Model loaded from {model_path}")

# === Prepare test dataset ===
test_dir = 'DataSplit/test'
test_gen = ImageDataGenerator(rescale=1./255)

test_data = test_gen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # IMPORTANT: don't shuffle so predictions align with labels
)

# === Predict ===
y_pred_probs = model.predict(test_data)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_data.classes
class_labels = list(test_data.class_indices.keys())

# === Print classification report ===
print("\n📊 Classification Report:")
report = classification_report(y_true, y_pred, target_names=class_labels)
print(report)

# === Show confusion matrix ===
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()
