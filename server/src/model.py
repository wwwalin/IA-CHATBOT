from pathlib import Path
import pickle

BASE_DIR = Path(__file__).resolve().parent
SERVER_DIR = BASE_DIR.parent
MODELS_DIR = SERVER_DIR / "models"




class ModeloIntencion:
    def __init__(self):
        
        with open(MODELS_DIR / "model.pkl", "rb") as f:
            self.model = pickle.load(f)

        with open(MODELS_DIR / "vectorizer.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)

       

    def predecir(self, texto):
        X = self.vectorizer.transform([texto])
        return self.model.predict(X)[0]