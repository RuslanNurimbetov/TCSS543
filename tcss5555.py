#!/usr/bin/python3
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import xml.etree.ElementTree as ET
import argparse
import random

# Path to the saved model
MODEL_PATH = "C:/Users/Н/Desktop/UW MSCS 4.0/TCSS 555 Machine Learning/data/model2.pth"

# Gender Prediction Model Class
class GenderPredictionModel(torch.nn.Module):
    def __init__(self, cnn_output_dim):
        super(GenderPredictionModel, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(cnn_output_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid()  # Binary classification
        )

    def forward(self, x):
        return self.fc(x)

# Function to load the model and additional data
def load_model():
    # Load the saved checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    
    # Retrieve model parameters
    try:
        cnn_output_dim = checkpoint['cnn_output_dim']
        scaler = checkpoint['scaler']
    except KeyError:
        raise KeyError("The checkpoint is missing required keys ('cnn_output_dim' or 'scaler'). Ensure the training script saves these properly.")

    # Initialize and load the model
    model = GenderPredictionModel(cnn_output_dim=cnn_output_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, scaler

# Function to preprocess the image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

# Function to predict gender
def predict_gender(model, cnn_model, scaler, image_tensor):
    with torch.no_grad():
        # Extract CNN features
        cnn_features = cnn_model(image_tensor).flatten().cpu().numpy()
        # Scale the features
        scaled_features = scaler.transform([cnn_features])
        # Convert to tensor for the gender model
        scaled_features_tensor = torch.tensor(scaled_features, dtype=torch.float32)
        # Predict gender
        output = model(scaled_features_tensor)
        prediction = (output > 0.5).float().item()
    return 'male' if prediction == 0 else 'female'

# Function to generate XML
def generate_xml(output_dir, image_name, gender):
    random_stats = {
        "age_group": random.choice(["xx-24", "25-34", "35-49", "50-xx"]),
        "extrovert": round(random.uniform(1.0, 5.0), 2),
        "neurotic": round(random.uniform(1.0, 5.0), 2),
        "agreeable": round(random.uniform(1.0, 5.0), 2),
        "conscientious": round(random.uniform(1.0, 5.0), 2),
        "open": round(random.uniform(1.0, 5.0), 2),
    }
    user = ET.Element("user")
    user.set("id", os.path.splitext(image_name)[0])
    user.set("age_group", random_stats["age_group"])
    user.set("gender", gender)
    user.set("extrovert", str(random_stats["extrovert"]))
    user.set("neurotic", str(random_stats["neurotic"]))
    user.set("agreeable", str(random_stats["agreeable"]))
    user.set("conscientious", str(random_stats["conscientious"]))
    user.set("open", str(random_stats["open"]))

    output_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}.xml")
    tree = ET.ElementTree(user)
    tree.write(output_path, encoding='utf-8', xml_declaration=True)

# Main function to process images
def process_images(input_dir, output_dir):
    # Load model, scaler, and CNN
    model, scaler = load_model()
    cnn_model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
    cnn_model.classifier = torch.nn.Identity()  # Remove fully connected layers
    cnn_model.eval()

    os.makedirs(output_dir, exist_ok=True)

    for image_name in os.listdir(input_dir):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, image_name)
            image_tensor = preprocess_image(image_path)
            gender = predict_gender(model, cnn_model, scaler, image_tensor)
            generate_xml(output_dir, image_name, gender)
            print(f"Processed {image_name}: Gender = {gender}")

# Command-line argument handling
def main():
    parser = argparse.ArgumentParser(description="Predict gender from images and generate XML files.")
    parser.add_argument("-i", "--input", required=True, help="Path to the input directory containing images.")
    parser.add_argument("-o", "--output", required=True, help="Path to the output directory for XML files.")
    args = parser.parse_args()

    process_images(args.input, args.output)

if __name__ == "__main__":
    main()