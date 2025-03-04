import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("100epoch.h5")

# Define labels
LABELS = {0: "Normal", 1: "Pneumonia"}

# Streamlit Page Configuration
st.set_page_config(page_title="Pneumonia Detection", layout="centered", page_icon="🩺")

# Custom Styling
st.markdown(
    """
    <style>
        .stApp {
            background: black;
        }
        .main-container {
            background-color: rgba(255, 255, 255, 0.95);
            padding: 20px;
            border-radius: 12px;
            box-shadow: 2px 4px 12px rgba(0, 0, 0, 0.2);
        }
        h1, h2, h3, h4 {
            color: #1e3a8a;
            text-align: center;
        }
        .stButton>button {
            background-color: #2563eb;
            color: white;
            font-size: 16px;
            padding: 10px;
            border-radius: 6px;
            border: none;
        }
        .stButton>button:hover {
            background-color: #1e40af;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and Instructions
st.title("🩺 Pneumonia Detection System")
st.markdown("### Upload a Chest X-ray Image for Diagnosis")

# Sidebar Instructions
with st.sidebar:
    st.header("How to Use")
    st.write("1. Upload a chest X-ray image.")
    st.write("2. View prediction results with confidence score.")
    st.write("3. Consult a medical professional for further evaluation.")
    st.warning("This tool is for informational purposes only.")

# File Uploader
uploaded_file = st.file_uploader("Upload Chest X-ray Image (JPEG/PNG)", type=["jpeg", "jpg", "png"])

def preprocess_image(image):
    """
    Preprocess the uploaded image for prediction.
    - Resize to model's input size (64x64)
    - Normalize pixel values
    - Convert grayscale images to RGB
    """
    image = image.resize((64, 64))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

# Prediction Logic
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded X-ray", use_column_width=True)
        st.write("---")
        
        with st.spinner("Analyzing the image..."):
            preprocessed_image = preprocess_image(image)
            prediction = model.predict(preprocessed_image)[0][0]
            predicted_class = LABELS[int(prediction >= 0.4)]
            probability = prediction if predicted_class == "Pneumonia" else 1 - prediction
        
        st.subheader("Prediction Result")
        st.write(f"**Predicted Condition:** `{predicted_class}`")
        st.write(f"**Confidence Score:** `{probability:.2f}`")
        
        if predicted_class == "Pneumonia":
            st.error("⚠️ This X-ray suggests possible Pneumonia. Please seek medical advice.")
        else:
            st.success("✅ The X-ray appears normal.")
        
        st.progress(int(probability * 100))
        
        with st.expander("🔍 Model Insights"):
            st.write("This model utilizes a fine-tuned deep learning approach for pneumonia detection.")
            st.code("model.summary()", language="python")
        
        if st.button("Reset"):
            st.experimental_rerun()
    except Exception as e:
        st.error("The uploaded file is not a valid image. Please upload a JPEG or PNG file.")
else:
    st.info("Upload an image to begin analysis.")

# Footer
st.markdown("""
    ---
    <div style="text-align: center;">
        Developed for research and educational purposes
    </div>
""", unsafe_allow_html=True)
