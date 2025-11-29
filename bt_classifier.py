import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from sklearn.model_selection import train_test_split

# Set dataset paths
dataset_path = "C:/Users/MBKEERTHIVASAN/Downloads/archive (1)"
categories = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Load and preprocess data
data, labels = [], []
for category in categories:
    path = os.path.join(dataset_path, 'training', category)
    class_num = categories.index(category)
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            if img_array is None:
                print(f"Failed to read image {img} in {category} folder.")
                continue
            resized_array = cv2.resize(img_array, (128, 128))
            data.append(resized_array)
            labels.append(class_num)
        except Exception as e:
            print(f"Failed to process image {img}: {e}")

data = np.array(data).reshape(-1, 128, 128, 1)
data = data / 255.0
labels = np.array(labels)
labels = tf.keras.utils.to_categorical(labels)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=32)

# Save the trained model
model.save('brain_tumor_classifier.h5')