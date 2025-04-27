import cv2
import os
import numpy as np
from tqdm import tqdm

def extract_frames(video_path, output_folder, sample_rate=30, max_frames=1000):
    """
    Extract frames from a video at the given sample rate
    
    Parameters:
    - video_path: Path to the video file
    - output_folder: Folder to save extracted frames
    - sample_rate: Save every nth frame
    - max_frames: Maximum number of frames to extract
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created directory: {output_folder}")
    
    # Open the video
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return 0
    
    # Get video properties
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    
    print(f"Video info:")
    print(f"- Total frames: {total_frames}")
    print(f"- FPS: {fps:.2f}")
    print(f"- Duration: {duration:.2f} seconds")
    print(f"- Extracting every {sample_rate}th frame")
    
    # Extract frames
    frame_count = 0
    saved_count = 0
    
    # Use tqdm for progress bar
    with tqdm(total=min(total_frames, max_frames * sample_rate)) as pbar:
        while True:
            success, frame = video.read()
            if not success or saved_count >= max_frames:
                break
                
            # Save frame at the specified sample rate
            if frame_count % sample_rate == 0:
                frame_path = os.path.join(output_folder, f"fish_frame_{saved_count:05d}.jpg")
                cv2.imwrite(frame_path, frame)
                saved_count += 1
                
            frame_count += 1
            pbar.update(1)
            
    video.release()
    print(f"\nExtracted {saved_count} frames to {output_folder}")
    return saved_count

def organize_frames_into_behaviors(frames_folder, behavior_name="swimming"):
    """
    Organize extracted frames into a behavior folder for training
    """
    # Create behavior directory
    behavior_dir = os.path.join(frames_folder, behavior_name)
    if not os.path.exists(behavior_dir):
        os.makedirs(behavior_dir)
        print(f"Created behavior directory: {behavior_dir}")
    
    # Get all frames
    frames = [f for f in os.listdir(frames_folder) if f.endswith('.jpg') and os.path.isfile(os.path.join(frames_folder, f))]
    
    # Move frames to behavior directory
    for frame in frames:
        src_path = os.path.join(frames_folder, frame)
        dst_path = os.path.join(behavior_dir, frame)
        
        # Only move if it's not already in a subdirectory
        if os.path.dirname(src_path) == frames_folder:
            os.rename(src_path, dst_path)
    
    print(f"Organized {len(frames)} frames into {behavior_name} behavior")

# Example usage
video_path = r"C:\Users\twish\Videos\4K Video Downloader+\Relaxing 2 minutes colourful fish aquarium video.mp4"
output_folder = r"fish_dataset"

# Extract frames
num_frames = extract_frames(
    video_path=video_path,
    output_folder=output_folder,
    sample_rate=15,  # Extract every 15th frame (adjust based on your video)
    max_frames=200   # Maximum 200 frames to extract
)

# If you want to label the frames as a particular behavior
# Uncomment the following line and specify the behavior
# organize_frames_into_behaviors(output_folder, "swimming")




# Fish Behavior Recognition Model
# Complete pipeline from data preparation to model evaluation

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import shutil
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Configuration parameters
IMG_SIZE = 224  # Input image size for the model
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 0.001

# Directories
BASE_DIR = "fish_dataset"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")
TEST_DIR = os.path.join(BASE_DIR, "test")
MODEL_PATH = "fish_behavior_model.h5"

def create_dataset_structure():
    """Create train/val/test splits from extracted frames"""
    print("Creating dataset structure...")
    
    # Create necessary directories
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)
    
    # Get behavior categories (subdirectories in BASE_DIR)
    behaviors = [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]
    
    if not behaviors:
        print("No behavior categories found!")
        print("Please organize your frames into behavior categories first.")
        print("Example: fish_dataset/swimming/, fish_dataset/feeding/, etc.")
        return False
    
    print(f"Found behavior categories: {behaviors}")
    
    # Create behavior subdirectories in train/val/test folders
    for behavior in behaviors:
        os.makedirs(os.path.join(TRAIN_DIR, behavior), exist_ok=True)
        os.makedirs(os.path.join(VAL_DIR, behavior), exist_ok=True)
        os.makedirs(os.path.join(TEST_DIR, behavior), exist_ok=True)
        
        # Get all image files for this behavior
        behavior_dir = os.path.join(BASE_DIR, behavior)
        images = [f for f in os.listdir(behavior_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if not images:
            print(f"Warning: No images found in {behavior_dir}")
            continue
            
        print(f"Splitting {len(images)} images for '{behavior}' behavior")
        
        # Shuffle images
        random.shuffle(images)
        
        # Split into train (70%), validation (15%), test (15%)
        train_split = int(0.7 * len(images))
        val_split = int(0.85 * len(images))
        
        train_images = images[:train_split]
        val_images = images[train_split:val_split]
        test_images = images[val_split:]
        
        # Copy images to respective directories
        for img in train_images:
            shutil.copy2(
                os.path.join(behavior_dir, img),
                os.path.join(TRAIN_DIR, behavior, img)
            )
            
        for img in val_images:
            shutil.copy2(
                os.path.join(behavior_dir, img),
                os.path.join(VAL_DIR, behavior, img)
            )
            
        for img in test_images:
            shutil.copy2(
                os.path.join(behavior_dir, img),
                os.path.join(TEST_DIR, behavior, img)
            )
        
        print(f"  Train: {len(train_images)}, Validation: {len(val_images)}, Test: {len(test_images)}")
    
    return True

def prepare_data_generators():
    """Create data generators for training and validation"""
    print("Setting up data generators...")
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        zoom_range=0.2
    )
    
    # Only rescaling for validation and test sets
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load data
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    validation_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, validation_generator, test_generator

def build_basic_model(num_classes):
    """Build a CNN model for fish behavior classification"""
    print("Building CNN model...")
    
    model = Sequential([
        # First convolutional block
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Second convolutional block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Third convolutional block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Flatten and fully connected layers
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def build_transfer_learning_model(num_classes):
    """Build a model based on MobileNetV2 with transfer learning"""
    print("Building MobileNetV2 transfer learning model...")
    
    # Load MobileNetV2 base model
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom top layers
    model = Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, train_generator, validation_generator):
    """Train the model with callbacks for best performance"""
    print("Training model...")
    
    # Set up callbacks
    callbacks = [
        # Save best model during training
        ModelCheckpoint(
            MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        # Stop training if no improvement
        EarlyStopping(
            monitor='val_accuracy',
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate when plateau is reached
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train the model
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def evaluate_model(model, test_generator):
    """Evaluate model on test set"""
    print("\nEvaluating model on test set...")
    
    # Get all test images and labels
    test_generator.reset()
    test_steps = len(test_generator)
    
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(test_generator, steps=test_steps)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Get predictions
    test_generator.reset()
    predictions = model.predict(test_generator, steps=test_steps)
    y_pred = np.argmax(predictions, axis=1)
    
    # True labels
    y_true = test_generator.classes[:len(y_pred)]
    
    # Get class names
    class_names = list(test_generator.class_indices.keys())
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    return test_accuracy, y_true, y_pred

def plot_training_history(history):
    """Plot training and validation accuracy/loss"""
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def real_time_prediction(model, class_names, webcam_source=0):
    """Run real-time prediction on webcam or video feed"""
    print("\nStarting real-time prediction...")
    print("Press 'q' to quit")
    
    # Open webcam or video file
    cap = cv2.VideoCapture(webcam_source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {webcam_source}")
        return
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
            
        # Prepare frame for prediction
        resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        normalized = resized / 255.0
        batch = np.expand_dims(normalized, axis=0)
        
        # Make prediction
        prediction = model.predict(batch, verbose=0)
        class_idx = np.argmax(prediction[0])
        confidence = prediction[0][class_idx]
        
        # Get predicted class name
        behavior = class_names[class_idx]
        
        # Display result on frame
        label = f"{behavior}: {confidence:.2f}"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Display the frame
        cv2.imshow('Fish Behavior Recognition', frame)
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

def main():
    """Main execution function"""
    print("Fish Behavior Recognition Model Training Pipeline")
    print("=" * 50)
    
    # Check if dataset structure exists, if not create it
    if not (os.path.exists(TRAIN_DIR) and os.path.exists(VAL_DIR) and os.path.exists(TEST_DIR)):
        success = create_dataset_structure()
        if not success:
            return
    
    # Prepare data generators
    train_generator, validation_generator, test_generator = prepare_data_generators()
    
    # Get number of classes
    num_classes = len(train_generator.class_indices)
    class_names = list(train_generator.class_indices.keys())
    print(f"Training model for {num_classes} behaviors: {class_names}")
    
    # Choose model type
    print("\nSelect model type:")
    print("1. Basic CNN")
    print("2. MobileNetV2 (Transfer Learning)")
    model_choice = input("Enter choice (1 or 2): ").strip()
    
    # Build model based on choice
    if model_choice == "2":
        model = build_transfer_learning_model(num_classes)
    else:
        model = build_basic_model(num_classes)
    
    # Print model summary
    model.summary()
    
    # Train the model
    history = train_model(model, train_generator, validation_generator)
    
    # Plot training history
    plot_training_history(history)
    
    # Load best model for evaluation
    best_model = load_model(MODEL_PATH)
    
    # Evaluate on test set
    test_accuracy, y_true, y_pred = evaluate_model(best_model, test_generator)
    
    # Ask if user wants to run real-time prediction
    run_realtime = input("\nDo you want to run real-time prediction? (y/n): ").strip().lower()
    if run_realtime == 'y':
        source_type = input("Use webcam (w) or video file (v)? ").strip().lower()
        
        if source_type == 'v':
            video_path = input("Enter video file path: ").strip()
            # Remove quotes if included
            video_path = video_path.strip('"\'')
            real_time_prediction(best_model, class_names, video_path)
        else:
            # Default to webcam (device 0)
            real_time_prediction(best_model, class_names)
    
    print("\nTraining and evaluation complete!")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Confusion matrix saved to: confusion_matrix.png")
    print(f"Training history saved to: training_history.png")

if __name__ == "__main__":
    main()

