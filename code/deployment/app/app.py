import streamlit as st
from streamlit_drawable_canvas import st_canvas
import requests
import numpy as np

# FastAPI endpoint
FASTAPI_URL = "http://fastapi:8000/predict"

# Set up canvas configuration
st.title("MNIST")

# Create a canvas component
canvas_result = st_canvas(
    stroke_width=40,
    stroke_color="#000000",
    background_color="#FFFFFF",
    update_streamlit=True,
    height=560,
    width=560,
    drawing_mode="freedraw",  # Other options: "line", "rect", "circle", "transform"
    key="canvas",
)
    

if st.button("Predict"):
    if canvas_result.image_data is not None:
        # Prepare the data for the API request
        # st.image(canvas_result.image_data)
        
        # Send a request to the FastAPI prediction endpoint
        scale = 20
        image = canvas_result.image_data
        # picture is grayscale by definition, so we can take only one channel
        image = 1 - np.array([image[i::scale, i::scale, 0] for i in range(scale)]).mean(axis=0) / 255
        print(image.shape)
        response = requests.post(FASTAPI_URL, json={'image': image.tolist()}, timeout=5)
        prediction = response.json()["prediction"]
        
        # Display the result
        st.success(f"The model predicts class: {prediction}")