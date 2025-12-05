import requests
from weather import UnifiedWeatherAgent
from plant import PlantIdentifierAgent


class VerdaAgent:
    def __init__(self, model_path, class_names):
        self.classifier = PlantIdentifierAgent(model_path, class_names)
        self.weather_agent = UnifiedWeatherAgent()
        self.last_identification = {}

    def identify_plant(self, image_path):
        result = self.classifier.predict(image_path)
        self.last_identification = result.copy()
        return result


class GenerativeVerdaAgent(VerdaAgent):
    def __init__(self, model_path, class_names):
        super().__init__(model_path, class_names)
        self.ollama_url = "http://localhost:11434/api/generate"
        self.model_name = "llama3.2"

    def chat_with_llm(self, prompt):
        """Send prompt to Ollama and return clean text."""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 150  # Limit tokens for eco-efficiency
                }
            }
            r = requests.post(self.ollama_url, json=payload, timeout=15)
            r.raise_for_status()
            return r.json().get("response", "").strip()
        except Exception as e:
            return f"Conseils non disponibles (LLM hors ligne)"

    # ---------------- IDENTIFICATION + CONSEILS ----------------
    def identify_plant(self, image_path, lang="fr"):
        result = super().identify_plant(image_path)

        plant_name = result.get("plant_name", "Inconnu")
        health_state = result.get("disease_status", "Inconnu")

        self.last_identification = {
            "plant_name": plant_name,
            "health_state": health_state,
            "confidence": result.get("confidence", 0)
        }

        # OPTIMIZED PROMPT: Use French names for better context
        prompt = (
            f"Plante: {plant_name}\n"
            f"État de santé: {health_state}\n\n"
            f"Donne EXACTEMENT 3 conseils écologiques pratiques pour cette situation. "
            f"Format:\n"
            f"1. [conseil pratique de 30-40 mots]\n"
            f"2. [conseil pratique de 30-40 mots]\n"
            f"3. [conseil pratique de 30-40 mots]\n\n"
            f"Sois direct, spécifique et pratique. Pas d'introduction ni de conclusion."
        )

        recommendation = self.chat_with_llm(prompt)

        return {
            "plant_name": plant_name,
            "disease_status": health_state,
            "confidence": result.get("confidence", 0),
            "recommendation": recommendation,
            "full_label": result.get("full_label", "")
        }

    # ---------------- CONSEILS BASÉS SUR LA MÉTÉO ----------------
    def recommend(self, lat, lon, lang="fr"):
        plant_name = self.last_identification.get("plant_name", "Inconnu")
        health_state = self.last_identification.get("health_state", "Inconnu")

        weather_yesterday = self.weather_agent.get_yesterday_weather(lat, lon)
        forecast_today = self.weather_agent.get_today_forecast(lat, lon)

        prompt = (
            f"Plante: {plant_name}\n"
            f"État: {health_state}\n"
            f"Météo: Hier={weather_yesterday}, Aujourd'hui={forecast_today}\n\n"
            f"Donne 3 conseils courts (max 30 mots chacun) adaptés à cette météo et cette plante."
        )

        recommendation = self.chat_with_llm(prompt)

        return {
            "plant_name": plant_name,
            "disease_status": health_state,
            "recommendation": recommendation,
            "weather": {
                "yesterday": weather_yesterday,
                "today": forecast_today
            }
        }