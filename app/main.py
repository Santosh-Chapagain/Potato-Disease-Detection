from pathlib import Path
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Project root (parent of 'app')
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = str(BASE_DIR / "potato_disease_model.h5")
STATIC_DIR = BASE_DIR / "app" / "static"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

print("Loading model from:", MODEL_PATH)  # Verify this prints correctly

# Verify file exists before loading
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

MODEL = tf.keras.models.load_model(MODEL_PATH, compile=False)


CLASS_NAMES = ["Early Blight", "Late_Blight", "Healthy"]


HOST = "127.0.0.1"
PORT = int(os.getenv("PORT", "8000"))


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.get("/")
async def home():
    return FileResponse(
        STATIC_DIR / "index.html",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image = read_file_as_image(await file.read())
    image_batch = np.expand_dims(image, 0)
    prediction = MODEL.predict(image_batch)

    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])

    return {
        "class": predicted_class,
        "confidence": float(confidence)
    }


@app.on_event("startup")
async def print_startup_links():
    # Always print direct links so they are visible even with minimal logging.
    print(f"Server URL: http://{HOST}:{PORT}")
    print(f"API Docs:   http://{HOST}:{PORT}/docs")


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="info", access_log=True)
