import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import pickle
import io

# Initialize FastAPI app
app = FastAPI()

# Load the model and labels
with open("skin_cancer_model-0.1.0.pkl", "rb") as model_file:
    model = pickle.load(model_file)

labels = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]


@app.get('/')
async def index():
    return {'message', 'Hello Ji'}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the image file
    image = Image.open(io.BytesIO(await file.read()))
    image = image.resize((32, 32))  # Resize to match model input
    image_array = np.asarray(image) / 255.0  # Normalize the image
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    # Make prediction
    prediction = model.predict(image_array)
    predicted_class = labels[np.argmax(prediction)]
    return prediction[0]


if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000)
