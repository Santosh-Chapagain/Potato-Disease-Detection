# Potato Disease Detection

A FastAPI web app for potato leaf disease prediction using a trained TensorFlow model.

## Features

- Upload a potato leaf image from the browser
- Predict disease class with confidence score
- Simple FastAPI backend with HTML frontend
- Model inference using `potato_disease_model.h5`

## Project Structure

- `app/main.py` - FastAPI backend and prediction API
- `app/static/index.html` - Frontend UI
- `potato_disease_model.h5` - Trained model file
- `requirements.txt` - Python dependencies
- `PlantVillage/` - Dataset folders

## Requirements

- Python 3.10+ (recommended)
- pip

## Setup

1. Create virtual environment:

```powershell
python -m venv venv
```

2. Activate virtual environment:

```powershell
.\venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Run the App

Start the server:

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open in browser:

- App: http://127.0.0.1:8000
- API docs: http://127.0.0.1:8000/docs

## API

### POST /predict

Upload one image file as form-data key `file`.

Response format:

```json
{
  "class": "Early Blight",
  "confidence": 0.98
}
```

## Notes

- Do not commit your `venv/` folder to Git.
- If frontend changes do not appear, hard refresh with `Ctrl+F5`.
- Ensure you are opening `http://127.0.0.1:8000` (same host/port as Uvicorn).

## License

This project is for educational and demonstration purposes.
