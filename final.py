import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("100epoch.h5")

# Define labels
LABELS = {0: "Normal", 1: "Pneumonia"}

# Set up Streamlit page config
st.set_page_config(page_title="Pneumonia Detection App", layout="centered", page_icon="🩺")

# Function to set a background image
def set_background(image_url):
    st.markdown(
        f"""
        <style>
            .stApp {{
                background-image: url({image_url});
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}
            .main-container {{
                background-color: rgba(255, 255, 255, 0.9);
                border-radius: 15px;
                padding: 20px;
                box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
            }}
            h1 {{
                color: #2E8B57;
                text-align: center;
                font-weight: bold;
            }}
            h2, h3, h4 {{
                color: #2E8B57;
                text-align: center;
            }}
            .stButton>button {{
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                padding: 10px;
                border-radius: 5px;
                border: none;
            }}
            .stButton>button:hover {{
                background-color: #2E7D32;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# Call the background setting function
set_background("https://t4.ftcdn.net/jpg/00/56/20/47/360_F_56204798_Zb7OA4nUEhjTsEEdQJ5Ce91AvAWSbAW1.jpg")

# Application title
st.title("🩺 Pneumonia Detection App")
st.markdown("### Upload a chest X-ray image to predict if it indicates **Normal** or **Pneumonia**.")

# Sidebar instructions
with st.sidebar:
    st.header("How to Use")
    st.write("1. Upload a Chest X-ray image.")
    st.write("2. View prediction results with confidence scores.")
    st.write("3. Consult a doctor for medical advice.")
    st.info("This tool is for educational purposes only.")

# File uploader
uploaded_file = st.file_uploader("Upload Chest X-ray Image (JPEG/PNG format)", type=["jpeg", "jpg", "png"])

def preprocess_image(image):
    """
    Preprocess the uploaded image for prediction.
    - Resize to the model's input size (64x64)
    - Normalize pixel values
    - Convert grayscale images to RGB by duplicating the channel.
    """
    image = image.resize((64, 64))  # Resize to (64, 64)
    if image.mode != 'RGB':
        image = image.convert('RGB')  # Convert grayscale to RGB
    image_array = np.array(image) / 255.0  # Normalize pixel values
    return np.expand_dims(image_array, axis=0)  # Add batch dimension

# Prediction logic
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("---")

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Make prediction
    prediction = model.predict(preprocessed_image)[0][0]
    predicted_class = LABELS[int(prediction >= 0.4)]
    probability = prediction if predicted_class == "Pneumonia" else 1 - prediction

    # Display prediction results
    st.subheader("Prediction Results")
    st.write(f"**Predicted Class:** `{predicted_class}`")
    st.write(f"**Confidence Score:** `{probability:.2f}`")

    # Add visual feedback with colors
    if predicted_class == "Pneumonia":
        st.error("⚠️ The X-ray suggests possible Pneumonia. Please consult a doctor.")
    else:
        st.success("✅ The X-ray appears Normal.")

    # Visualize prediction probability with a progress bar
    st.progress(int(probability * 100))

    # Add option to view model details
    with st.expander("🔍 View Model Details"):
        st.write("This model uses a pre-trained VGG19 architecture fine-tuned for Pneumonia detection.")
        st.code("model.summary()", language="python")

    # Option to re-analyze
    if st.button("Analyze Again"):
        st.experimental_rerun()
else:
    st.info("Please upload an image to start the analysis.")

# Footer
st.markdown("""
    ---
    <div style="text-align: center;">
         <a href="https://your-portfolio-link.com" target="_blank">Neha Bhawthankar</a>
    </div>
""", unsafe_allow_html=True)