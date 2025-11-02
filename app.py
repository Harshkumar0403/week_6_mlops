import os
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import storage

# ==========================================================
# CONFIG
# ==========================================================
SERVICE_ACCOUNT_KEY = "github-dvc-key.json"  # must be in repo/VM
BUCKET_NAME = "mlops-course-phonic-axle-473506-u8-unique"
# this is where DVC stored your model in GCS
GCS_MODEL_BLOB = "dvcstore/files/md5/b5/0729b0bfd352c510335b6e0c71b236"
LOCAL_MODEL_PATH = Path("models/model.pkl")

app = FastAPI(
    title="Iris Species Prediction API",
    description="Predict iris species using model downloaded from GCS",
    version="1.0.0",
)

model = None  # will be loaded at startup


# ==========================================================
# INPUT SCHEMA
# ==========================================================
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# ==========================================================
# HELPERS
# ==========================================================
def download_model_from_gcs():
    """
    Downloads the model from GCS to local models/model.pkl
    """
    # make sure models/ exists
    LOCAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # set creds for google client
    if os.path.exists(SERVICE_ACCOUNT_KEY):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_KEY
    else:
        raise FileNotFoundError(
            f"Service account JSON '{SERVICE_ACCOUNT_KEY}' not found in current dir."
        )

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(GCS_MODEL_BLOB)
    blob.download_to_filename(LOCAL_MODEL_PATH)
    print("✅ Model downloaded from GCS →", LOCAL_MODEL_PATH)


def load_model():
    """
    Ensures model is present locally and loads it with joblib.
    """
    if not LOCAL_MODEL_PATH.exists():
        print("⏳ model.pkl not found locally. Downloading from GCS...")
        download_model_from_gcs()

    mdl = joblib.load(LOCAL_MODEL_PATH)
    print("✅ Model loaded into memory.")
    return mdl


# ==========================================================
# STARTUP
# ==========================================================
@app.on_event("startup")
def on_startup():
    global model
    try:
        model = load_model()
    except Exception as e:
        # don’t crash the app, but log the error
        print(f"⚠️ Could not load model on startup: {e}")
        model = None


# ==========================================================
# ROUTES
# ==========================================================
@app.get("/")
def root():
    return {"message": "Iris Species Prediction API is running."}


@app.post("/predict")
def predict(features: IrisInput):
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Try again later.")

    X = [[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width,
    ]]

    try:
        y_pred = model.predict(X)[0]

        # If model returns string labels like "setosa"
        if isinstance(y_pred, str):
            species = y_pred
        else:
            label_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
            species = label_map.get(int(y_pred), "unknown")

        return {
            "status": "success",
            "predicted_label": y_pred,
            "species": species
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")



# ==========================================================
# MAIN (so you can run: python app.py)
# ==========================================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # nice for local/dev
    )

