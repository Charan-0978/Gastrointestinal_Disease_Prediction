import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

def load_data(data_dir, img_size=(128, 128)):
    X = []
    y = []
    
    class_names = sorted(os.listdir(data_dir))

    for label, class_name in enumerate(class_names):
        class_path = os.path.join(data_dir, class_name)
        
        # Skip if it's not a directory (like .DS_Store or hidden files)
        if not os.path.isdir(class_path):
            continue

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)

            # Skip non-image files
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.resize(img, img_size)
                X.append(img)
                y.append(label)
            except Exception as e:
                print(f"Skipping image {img_path}: {e}")
                continue

    X = np.array(X, dtype='float32') / 255.0
    y = to_categorical(y, num_classes=len(class_names))

    return train_test_split(X, y, test_size=0.2, random_state=42), class_names
