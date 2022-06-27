import cv2
import time
import tempfile
import argparse
import streamlit as st
import onnxruntime as ort
from PIL import Image
from helpers import predict_image

# Set Argument Parse
parser = argparse.ArgumentParser()
parser.add_argument(
    "-d",
    "--device",
    type=str,
    default="CPU",
    help="Device to use for inference (CPU or CUDA)",
)
parser.add_argument(
    "-",
    "--model",
    type=str,
    default="./saved_model-inanimate_liveness.onnx",
    help="Path to model",
)
value_parser = parser.parse_args()

# Define constant variable
DEVICE_INFERENCE = value_parser.device
MODEL_PATH = value_parser.model

# Set page config
st.set_page_config(
    page_title="Inanimate Face Liveness",
    page_icon="🤖",
)

# Load model
@st.cache(allow_output_mutation=True)
def load_model(model_path, device_inference="cpu"):
    if device_inference.lower() == "cpu":
        ort_session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
    elif device_inference.lower() == "cuda":
        ort_session = ort.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider"],
        )
    else:
        st.error("Please select between CPU or CUDA!")
        st.stop()

    return ort_session


# Run load model
model = load_model(MODEL_PATH, DEVICE_INFERENCE)

# Main page
st.title("Inanimate Face Liveness")
st.write(
    """
        A lot of applications now use artificial intelligence like face recognition. Face recognition is a model, to detect if a face is the same or not. When we use face recognition models, we need to make sure that the faces are not animate. The purpose of this project is to detect if a face is inanimate or animate. Created with Res2Next50 models and some robust augmentations.
"""
)
st.markdown("  ")

uploaded_file = st.file_uploader("Upload image file", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    uploaded_file = Image.open(uploaded_file)
    st.markdown("  ")
    st.write("Source Image")
    st.image(uploaded_file)

    predict_button = st.button("Detect inanimate face")
    st.markdown("  ")

    if predict_button:
        with st.spinner("Wait for it..."):
            start_time = time.time()
            predicted_classes, predicted_score = predict_image(uploaded_file, model)
            st.write(f"Score: {predicted_score:.3f}")
            st.write(f"Classes: {predicted_classes}")
            st.write(f"Inference time: {(time.time() - start_time):.3f} seconds")
