import torch
from torchvision import transforms
from mobilenet_inference import MobileNetV2Classifier
from PIL import Image
import os


class PlantIdentifierAgent:
    def __init__(self, model_path, class_names):
        # Select device: GPU if available, else CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model and load weights
        self.model = MobileNetV2Classifier(num_classes=len(class_names)).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()  # inference mode

        # Class names
        self.class_names = class_names

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # French translations for plant names
        self.plant_translations = {
            'Apple': 'Pommier',
            'Bell_pepper': 'Poivron',
            'Cherry': 'Cerisier',
            'Corn_maize': 'Maïs',
            'Corn': 'Maïs',
            'Grape': 'Vigne',
            'Peach': 'Pêcher',
            'Potato': 'Pomme de terre',
            'Strawberry': 'Fraisier',
            'Tomato': 'Tomate'
        }
        
        # French translations for diseases
        self.disease_translations = {
            'healthy': 'Sain',
            'apple_scab': 'Tavelure du pommier',
            'black_rot': 'Pourriture noire',
            'cedar_apple_rust': 'Rouille grillagée',
            'bacterial_spot': 'Tache bactérienne',
            'powdery_mildew': 'Oïdium',
            'cercospora_leaf_spot': 'Cercosporiose',
            'common_rust': 'Rouille commune',
            'northern_leaf_blight': 'Brûlure des feuilles du Nord',
            'esca_(black_measles)': 'Esca (rougeole noire)',
            'leaf_blight': 'Brûlure des feuilles',
            'early_blight': 'Mildiou précoce',
            'late_blight': 'Mildiou tardif',
            'leaf_scorch': 'Brûlure des feuilles',
            'leaf_mold': 'Moisissure des feuilles',
            'septoria_leaf_spot': 'Septoriose',
            'yellow_leaf_curl_virus': 'Virus de l\'enroulement des feuilles'
        }

    def translate_to_french(self, plant_name, disease_status):
        """Translate plant and disease names to French"""
        # Clean up the names
        plant_clean = plant_name.strip().replace('_', ' ')
        disease_clean = disease_status.strip().lower().replace(' ', '_')
        
        # Translate plant
        french_plant = self.plant_translations.get(plant_name, plant_clean)
        
        # Translate disease
        french_disease = self.disease_translations.get(disease_clean, disease_status)
        
        return french_plant, french_disease

    def predict(self, image_path):
        image = Image.open(image_path).convert('RGB')
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            top_prob, top_class = torch.max(probs, dim=0)

            full_label = self.class_names[top_class.item()]
            if "___" in full_label:
                plant_name, disease_status = full_label.split("___", 1)
            else:
                plant_name = full_label
                disease_status = "Unknown"
        
        # Translate to French
        french_plant, french_disease = self.translate_to_french(plant_name, disease_status)

        return {
            "plant_name": french_plant,
            "plant_name_en": plant_name,
            "disease_status": french_disease,
            "disease_status_en": disease_status,
            "full_label": full_label,
            "confidence": top_prob.item()
        }


if __name__ == "__main__":
    model_path = os.path.join("checkpoints", "fine_tuned_mobilenet.pth")
    class_names = [
        'Apple___apple_scab', 'Apple___black_rot', 'Apple___cedar_apple_rust',
        'Apple___healthy', 'Bell_pepper___bacterial_spot', 'Bell_pepper___healthy',
        'Cherry___healthy', 'Cherry___powdery_mildew', 'Corn_maize___cercospora_leaf_spot',
        'Corn_maize___common_rust', 'Corn_maize___healthy', 'Corn_maize___northern_leaf_blight',
        'Grape___black_rot', 'Grape___esca_(black_measles)', 'Grape___healthy',
        'Grape___leaf_blight', 'Peach___bacterial_spot', 'Peach___healthy',
        'Potato___early_blight', 'Potato___healthy', 'Potato___late_blight',
        'Strawberry___healthy', 'Strawberry___leaf_scorch ', 'Tomato___bacterial_spot',
        'Tomato___early_blight', 'Tomato___healthy', 'Tomato___late_blight',
        'Tomato___leaf_mold', 'Tomato___septoria_leaf_spot', 'Tomato___yellow_leaf_curl_virus'
    ]

    agent = PlantIdentifierAgent(model_path, class_names)

    image_path = "D:/green-it-verda/backend/test_images/sample.jfif"
    result = agent.predict(image_path)

    print(f"Predicted: {result['full_label']}")
    print(f"Plant Name (FR): {result['plant_name']}")
    print(f"Plant Name (EN): {result['plant_name_en']}")
    print(f"Health Status (FR): {result['disease_status']}")
    print(f"Health Status (EN): {result['disease_status_en']}")
    print(f"Confidence: {result['confidence']:.2%}")