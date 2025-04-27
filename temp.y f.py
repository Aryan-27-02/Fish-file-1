import cv2
import os
import numpy as np

def extract_frames(video_path, output_folder, sample_rate=1):
    """Extract frames from a video at the given sample rate"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    video = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0
    
    while True:
        success, frame = video.read()
        if not success:
            break
            
        # Save frame at the specified sample rate
        if frame_count % sample_rate == 0:
            frame_path = os.path.join(output_folder, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_count += 1
            
        frame_count += 1
        
    video.release()
    return saved_count

# Example usage
video_file = "fish_behaviors/swimming.mp4"
output_dir = "dataset/swimming"
frames = extract_frames(video_file, output_dir, sample_rate=5)  # Save every 5th frame
print(f"Extracted {frames} frames")

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define image dimensions and other parameters
img_width, img_height = 224, 224
batch_size = 32
num_classes = 4  # Number of fish behavior categories

# Create data generators with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# Load training data
train_generator = train_datagen.flow_from_directory(
    'dataset',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

# Load validation data
validation_generator = train_datagen.flow_from_directory(
    'dataset',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Create the model
model = Sequential([
    # First convolutional block
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    
    # Second convolutional block
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    # Third convolutional block
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    # Fully connected layers
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),  # Prevent overfitting
    Dense(num_classes, activation='softmax')  # Output layer
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=25,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Save the model
model.save('fish_behavior_model.h5')


import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report

# Load the trained model
model = load_model('fish_behavior_model.h5')

# Evaluate on validation data
validation_generator.reset()
y_pred = model.predict(validation_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

# Get true classes
validation_generator.reset()
y_true = validation_generator.classes

# Print classification report
print("\nClassification Report:")
class_names = list(validation_generator.class_indices.keys())
print(classification_report(y_true, y_pred_classes, target_names=class_names))

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.tight_layout()
plt.savefig('training_history.png')
plt.show()



import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

def preprocess_frame(frame, target_size):
    """Preprocess a video frame for the model"""
    # Resize frame
    resized = cv2.resize(frame, target_size)
    # Normalize
    normalized = resized / 255.0
    # Expand dimensions for batch
    expanded = np.expand_dims(normalized, axis=0)
    return expanded

# Load the trained model
model = load_model('fish_behavior_model.h5')
class_names = ['feeding', 'resting', 'schooling', 'swimming']  # Update with your classes
target_size = (224, 224)

# Open webcam or video file
video_source = 0  # Use 0 for webcam or provide a video file path
cap = cv2.VideoCapture(video_source)

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Preprocess the frame
    processed_frame = preprocess_frame(frame, target_size)
    
    # Make prediction
    prediction = model.predict(processed_frame)
    predicted_class = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class]
    
    # Display result on frame
    label = f"{class_names[predicted_class]}: {confidence:.2f}"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Display the frame
    cv2.imshow('Fish Behavior Detection', frame)
    
    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D

# Create base model
base_model = MobileNetV2(weights='imagenet', include_top=False, 
                         input_shape=(img_width, img_height, 3))

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Create new model on top
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])





from tensorflow.keras.layers import TimeDistributed, LSTM

# For sequence data
def build_lstm_model(sequence_length, img_width, img_height, num_classes):
    # CNN feature extractor
    cnn_model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten()
    ])
    
    # Full sequence model
    model = Sequential([
        TimeDistributed(cnn_model, input_shape=(sequence_length, img_width, img_height, 3)),
        LSTM(256, return_sequences=False),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    return model