from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
# inside Docker it will be visible
from model_cnn import ModelCNN, transform
import torch
from torchvision import transforms

model = ModelCNN()
# inside Docker it will be visible
model.load_state_dict(torch.load("./mnist_cnn.pt", weights_only=True))

# Define the FastAPI app
app = FastAPI()

# Define the input data schema
class MNISTInput(BaseModel):
    image: list

# Define the prediction endpoint
@app.post("/predict")
def predict(input_data: MNISTInput):
    with torch.no_grad():
        data = np.array(input_data.image).astype('float32')
        data = transform(data).reshape([1, 1, 28, 28])
        print(data)
        print(data.shape)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        print(pred)
        return {"prediction": int(pred[0])}