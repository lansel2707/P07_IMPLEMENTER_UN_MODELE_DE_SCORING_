import mlflow
import mlflow.sklearn
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import json  # Pour la sauvegarde du JSON d'exemple

MLFLOW_TRACKING_URI = "http://localhost:5000"
MODEL_NAME = "Projet_Scoring_Best_Model"
MODEL_STAGE = "None"  # ou "Production", "Staging", "None" => latest

app = FastAPI(
    title="API de Scoring (MLflow Registry)",
    description="API qui interroge le modèle depuis MLflow Model Registry",
    version="1.0"
)

class InputData(BaseModel):
    data: dict

def get_features_from_signature(model_uri):
    import mlflow.models
    model_info = mlflow.models.get_model_info(model_uri)
    if model_info is not None and model_info.signature is not None:
        features = [col.name for col in model_info.signature.inputs.inputs]
        return features
    # Fallback: lire feature_names_in_ du modèle sklearn
    loaded_model = mlflow.sklearn.load_model(model_uri)
    if hasattr(loaded_model, "feature_names_in_"):
        return list(loaded_model.feature_names_in_)
    return []

def save_swagger_json(features, path="input_example_swagger.json"):
    example = {feat: 0 for feat in features}
    with open(path, "w") as f:
        json.dump({"data": example}, f, indent=2)
    print(f"\n\nExemple d'input JSON sauvegardé dans {path} (prêt à copier-coller dans Swagger):\n")
    print(json.dumps({"data": example}, indent=2))

def load_model_and_features():
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if MODEL_STAGE.lower() != "none":
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    else:
        model_uri = f"models:/{MODEL_NAME}/latest"
    print(f"Chargement du modèle depuis MLflow : {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)
    features = get_features_from_signature(model_uri)
    print(f"Features détectées automatiquement : {features}")
    if not features:
        raise Exception("Impossible de détecter automatiquement les features du modèle MLflow")
    save_swagger_json(features)
    return model, features

model, FEATURES = load_model_and_features()

@app.post("/predict/")
def predict(input: InputData):
    input_data = input.data
    missing = [feat for feat in FEATURES if feat not in input_data]
    if missing:
        return {"error": f"Features manquantes : {missing}. Attendues : {FEATURES}"}
    X = np.array([[input_data[feat] for feat in FEATURES]])
    proba = model.predict_proba(X)[0][1]
    pred = int(model.predict(X)[0])
    return {
        "prediction": pred,
        "probability_bad_payer": float(proba)
    }

@app.get("/")
def home():
    return {"message": "API de scoring – modèle appelé depuis MLflow Registry (FastAPI)"}

@app.get("/features/")
def get_features():
    return {"features": FEATURES}
