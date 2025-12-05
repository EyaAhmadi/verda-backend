import subprocess
import json

class LLMRecommender:
    def __init__(self, model="llama3.2"):
        self.model = model

    def ask(self, prompt):
        """Calls Ollama locally and returns the generated text."""
        try:
            result = subprocess.run(
                ["ollama", "run", self.model],
                input=prompt.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            return result.stdout.decode("utf-8").strip()

        except Exception as e:
            return f"[LLM Error] {e}"
