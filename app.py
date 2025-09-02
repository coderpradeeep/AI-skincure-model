import uvicorn
import pickle
import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io

app = FastAPI()

# Load your trained skin disease classification model
with open("skin_cancer_model-0.1.0.pkl", "rb") as f:
    model = pickle.load(f)


@app.get("/")
def index():
    return {"message": "Skin Disease Detection API is running!"}


@app.post("/predict")
async def predict_skin_disease(file: UploadFile = File(...)) -> str:
    # Read image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Preprocess image (make sure this matches your training preprocessing)
    image = image.resize((128, 128))  # Example size, adjust to your training size
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # (1, H, W, C)

    # Make prediction
    prediction = model.predict(image_array)  # depends on your model type
    # predicted_class_idx = np.argmax(prediction, axis=1)[0]

    return {"predicted_disease": prediction}


# Run app: uvicorn app:app --reload
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)