import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image


# Load the trained model
model = tf.keras.models.load_model("100epoch.h5")

# Define labels
LABELS = {0: "Normal", 1: "Pneumonia"}

# Set up Streamlit page config
st.set_page_config(page_title="Pneumonia Detection App", layout="wide", page_icon="🩺")

# Function to set a terracotta, beige, and teal color theme with responsive design
def set_responsive_theme():
    st.markdown(
        """
        <style>
            /* Full-width styling for responsiveness */
            .stApp {
                background-color: #F5EFE6; /* Light beige background */
                color: #5D5C61; /* Muted teal for text */
                max-width: 100%;
                padding: 0;
                margin: 0;
            }

            /* Main content styling */
            .main-container {
                background-color: #FFFFFF;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.1);
                width: 95%;
                margin: 0 auto;
            }

            /* Terracotta button styling */
            .stButton>button {
                width: 100%;
                background-color: #A34A28; /* Terracotta red */
                color: #FFFFFF;
                font-size: 18px;
                padding: 15px;
                border-radius: 5px;
                border: none;
            }

            .stButton>button:hover {
                background-color: #923E24;
            }

            /* Sidebar styling */
            .css-1d391kg {
                background-color: #A3C6C4 !important; /* Muted teal */
                color: #5D5C61;
            }

            /* Titles and headers */
            h1, h2, h3, h4 {
                color: #A34A28; /* Terracotta for headings */
                text-align: center;
            }

            /* Image styling */
            .stImage {
                display: block;
                margin: 20px auto;
                border: 3px solid #A3C6C4; /* Muted teal border */
                border-radius: 8px;
            }

            /* Progress bar styling */
            .stProgress {
                width: 100%;
                margin: 10px 0;
                background-color: #F5EFE6; /* Light beige */
            }

            /* Footer styling */
            footer {
                text-align: center;
                color: #5D5C61; /* Muted teal */
                margin-top: 30px;
                font-size: 14px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Call the theme setting function
set_responsive_theme()

# Application title
st.title("🩺Pneumonia Detection App🩺")
st.markdown(
    "### Upload a chest X-ray image to predict if it indicates **Normal** or **Pneumonia**."
)

# Sidebar instructions
with st.sidebar:
    st.markdown(
        """
        <style>
            .sidebar .sidebar-content h1, .sidebar .sidebar-content h2, .sidebar .sidebar-content h3 {
                font-size: 40px;  /* Increase header font size */
            }
        </style>
        """, unsafe_allow_html=True
    )
    st.header("👨‍⚕️ How to Use")
    st.write(
        """
        1. **🖼️ Upload a Chest X-ray Image**: 
        Choose a clear JPEG/PNG image for analysis.

        2. **🔍 View Prediction Results**: 
        The model will predict if the X-ray shows **Normal** 🟢 or **Pneumonia** 🟠, with a confidence score.

        3. **⚠️ Consult a Doctor**: 
        This tool **does not replace professional medical advice**. Always consult a healthcare provider for an accurate diagnosis, especially if the model suggests **Pneumonia** 🩺.

        4. **💡 Important Note**: 
        Even if the result is **Normal** 🟢, please visit a doctor if you experience symptoms like cough 🤧, fever 🌡️, or difficulty breathing 😷.
        """
    )
    st.info("This tool is for educational purposes only. Seek professional consultation for accurate diagnosis.")


# File uploader
uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image (JPEG/PNG format)", type=["jpeg", "jpg", "png"]
)

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
    st.image(image, caption="Uploaded Image", use_container_width=True)

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
        st.error("⚠️ The X-ray suggests possible **Pneumonia**. Please consult a doctor for a professional diagnosis and further steps.")

    else:
        st.success("✅ The X-ray appears **Normal**. However, if you have symptoms or concerns, it's always best to consult a doctor for further advice.")

    # Visualize prediction probability with a progress bar
    st.progress(int(probability * 100))

# Option to re-analyze
# if st.button("Analyze Again"):
#     # st.experimental_set_query_params()
#     # st.experimental_rerun()
#     # st.session_state.clear()  # Clears the session state, effectively "resetting" the app
#     # st.experimental_rerun()
#     st.rerun()

# Add option to view model details
with st.expander("🔍 View Model Details"):
    summary_string = []
    model.summary(print_fn=lambda x: summary_string.append(x))
    model_summary = "\n".join(summary_string)
    st.code(model_summary)

# Footer
st.markdown(
    """
    ---
    <footer>© 2025 Pneumonia Detection App. All rights reserved.</footer>
    """,
    unsafe_allow_html=True,
)
