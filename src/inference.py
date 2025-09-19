# inference.py
"""
Inference script for Monkey Species Classification
-------------------------------------------------
Usage:
    python inference.py --image path/to/image.jpg
"""

import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Define image size (same as training input for VGG16)
IMG_SIZE = (224, 224)

def load_and_preprocess(img_path):
    """Load and preprocess a single image for prediction."""
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # normalize
    return img_array

def predict_species(model_path, img_path, class_names):
    """Load model and predict the monkey species."""
    model = load_model(model_path)
    img_array = load_and_preprocess(img_path)
    preds = model.predict(img_array)[0]
    
    predicted_idx = np.argmax(preds)
    predicted_class = class_names[predicted_idx]
    confidence = preds[predicted_idx] * 100
    
    return predicted_class, confidence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monkey Species Classifier Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    args = parser.parse_args()

    # Monkey species list (replace with your dataset labels)
    class_names = [
        "Mantled Guereza",
        "Patas Monkey",
        "Bald Uakari",
        "Japanese Macaque",
        "Pygmy Marmoset",
        "White-headed Capuchin",
        "Silvery Marmoset",
        "Common Squirrel Monkey",
        "Black-headed Night Monkey",
        "Nilgiri Langur"
    ]

    model_path = "monkey_species_classifier.h5"
    
    species, confidence = predict_species(model_path, args.image, class_names)
    print(f"Predicted Species: {species}")
    print(f"Confidence: {confidence:.2f}%")
