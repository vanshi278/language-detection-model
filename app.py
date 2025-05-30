import streamlit as st
import pickle
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="Language Detection Tool",
    page_icon="🌍",
    layout="centered"
)

# Define model path
MODEL_PATH = Path(__file__).parent / 'model.pckl'

# Load model with error handling
@st.cache_resource
def load_model():
    try:
        with open(MODEL_PATH, 'rb') as file:
            model = pickle.load(file)
            return model
    except FileNotFoundError:
        st.error("❌ Model file not found. Please ensure 'model.pckl' exists in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

# Initialize model
try:
    Lrdetect_Model = load_model()
except Exception as e:
    st.error(f"❌ Failed to initialize model: {str(e)}")
    st.stop()

# UI Elements
st.title("🌍 Language Detection Tool")
st.write("Enter text to detect its language")

input_test = st.text_area(
    "Text Input",
    value='Hello my name is Samarth.',
    height=100,
    key="text_input"
)

if st.button("🔍 Detect Language", type="primary"):
    if input_test.strip():
        try:
            prediction = Lrdetect_Model.predict([input_test])
            st.success(f"✨ Detected Language: {prediction[0]}")
        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")
    else:
        st.warning("⚠️ Please enter some text to detect language")