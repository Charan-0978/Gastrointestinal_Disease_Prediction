from data_preprocessing import load_data
from model_cnn import build_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Load data
(X_train, X_val, y_train, y_val), class_names = load_data("kvasir-dataset")

# Build model
model = build_model((128, 128, 3), len(class_names))

# Callbacks
checkpoint = ModelCheckpoint("model.h5", save_best_only=True)
early_stop = EarlyStopping(patience=5, restore_best_weights=True)

# Train
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=15,
          batch_size=32, callbacks=[checkpoint, early_stop])


from sklearn.metrics import accuracy_score
import numpy as np

# Evaluate model on validation set
loss, accuracy = model.evaluate(X_val, y_val)
print("Validation Accuracy:", round(accuracy * 100, 2), "%")

# Optional: Custom accuracy calculation
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_val, axis=1)
custom_acc = accuracy_score(y_true, y_pred_classes)
print("Custom Accuracy:", round(custom_acc * 100, 2), "%")
