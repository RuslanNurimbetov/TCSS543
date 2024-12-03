#!/usr/bin/python3

import os
import argparse
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.metrics import RootMeanSquaredError
import random

# ----------------------------- #
# Function Definitions
# ----------------------------- #

def load_image(img_id, img_dir, img_size=(224, 224)):
    """
    Load and preprocess an image by resizing and normalizing it.
    """
    path = os.path.join(img_dir, f"{img_id}.jpg")
    img = Image.open(path).resize(img_size).convert('RGB')
    return np.array(img) / 255.0  # Normalize to [0, 1]

def save_to_XML(predictions, output_dir, test_ids, true_labels):
    """
    Save predictions into individual XML files for each user.
    """
    os.makedirs(output_dir, exist_ok=True)
    for userid, true_gender, pred in zip(test_ids, true_labels, predictions):
        user = ET.Element("user")
        user.set("id", str(userid))
        user.set("age_group", "xx-24" if random.random() > 0.5 else "25-34")  # Placeholder logic
        user.set("gender", "male" if pred['gender'] < 0.5 else "female")
        user.set("extrovert", f"{pred['extrovert']:.2f}")
        user.set("neurotic", f"{pred['neurotic']:.2f}")
        user.set("agreeable", f"{pred['agreeable']:.2f}")
        user.set("conscientious", f"{pred['conscientious']:.2f}")
        user.set("open", f"{pred['open']:.2f}")

        file_name = os.path.join(output_dir, f"{userid}.xml")
        tree = ET.ElementTree(user)
        tree.write(file_name, encoding='utf-8', xml_declaration=True)
    print(f"XML files have been saved in '{output_dir}'.")

# ----------------------------- #
# Main Script
# ----------------------------- #

def main(input_dir, output_dir):
    # ----------------------------- #
    # 1. Load Data
    # ----------------------------- #
    print("Loading data...")
    img_dir = os.path.join(input_dir, "image")
    csv_path = os.path.join(input_dir, "profile", "profile.csv")
    
    data = pd.read_csv(csv_path)
    data["image"] = data["userid"].apply(lambda x: load_image(x, img_dir))
    X = np.stack(data["image"].values).astype('float32')  # Ensure all images are float32
    y = np.array(data["gender"].values, dtype='float32')  # Convert labels to float32
    user_ids = data["userid"].values  # User IDs for XML generation

    # Split into training and validation sets
    X_train, X_val, y_train, y_val, user_ids_train, user_ids_val = train_test_split(
        X, y, user_ids, test_size=0.2, random_state=42
    )

    # ----------------------------- #
    # 2. Define CNN Model
    # ----------------------------- #
    print("Defining model...")
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(512, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Sigmoid activation for binary classification
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', RootMeanSquaredError()])

    # ----------------------------- #
    # 3. Data Augmentation
    # ----------------------------- #
    print("Creating data generators...")
    train_datagen = ImageDataGenerator(
        horizontal_flip=True,
        rotation_range=20,
        brightness_range=[0.8, 1.2],
        zoom_range=0.1
    )
    val_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow(X_train, y_train, batch_size=32)
    val_generator = val_datagen.flow(X_val, y_val, batch_size=32)

    # ----------------------------- #
    # 4. Train the Model
    # ----------------------------- #
    print("Training the model...")
    model.fit(train_generator, validation_data=val_generator, epochs=10)

    # ----------------------------- #
    # 5. Evaluate and Predict
    # ----------------------------- #
    print("Evaluating the model...")
    val_loss, val_accuracy, val_rmse = model.evaluate(val_generator)
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation RMSE: {val_rmse:.4f}")

    print("Making predictions on the validation set...")
    gender_predictions = model.predict(X_val).flatten()

    # Mock predictions for personality traits (placeholder logic)
    predictions = []
    for _ in range(len(gender_predictions)):
        predictions.append({
            "gender": gender_predictions[_],
            "extrovert": random.uniform(1.0, 5.0),
            "neurotic": random.uniform(1.0, 5.0),
            "agreeable": random.uniform(1.0, 5.0),
            "conscientious": random.uniform(1.0, 5.0),
            "open": random.uniform(1.0, 5.0)
        })

    # ----------------------------- #
    # 6. Save Predictions as XML
    # ----------------------------- #
    print("Saving predictions to XML...")
    save_to_XML(predictions, output_dir, user_ids_val, y_val)

# ----------------------------- #
# Entry Point
# ----------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CNN model and save predictions to XML.")
    parser.add_argument("-i", "--input", required=True, help="Path to the input directory")
    parser.add_argument("-o", "--output", required=True, help="Path to the output directory")
    args = parser.parse_args()

    main(args.input, args.output)
