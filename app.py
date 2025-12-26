# app.py
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
from PIL import Image
import logging

# ============================
# Page Config
# ============================
st.set_page_config(
    page_title="Sign Language Classifier",
    page_icon="✋",
    layout="centered"
)

# ============================
# Logger (minimal for deployment)
# ============================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================
# Title & Description
# ============================
st.title("✋ Real-Time Sign Language Classifier")
st.markdown("""
This app detects and classifies hand signs (A-Z, 0-9) in real-time using your webcam  
or from uploaded images.

**How to use:**
- Click **"Start Webcam"** to begin live detection
- Show clear hand signs in good lighting
- Or upload an image below for single prediction
""")

# ============================
# File Paths (Relative - Safe for Deployment)
# ============================
MODEL_PATH = "models/best_model.p"          # Place your model file in /models folder
LABELS_PATH = "models/labels.p"             # Place your label encoder here

# ============================
# Load Model & Labels
# ============================
@st.cache_resource
def load_model_and_labels():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}")
        st.info("Upload your 'best_model.p' and 'labels.p' files in a 'models' folder.")
        st.stop()
    
    if not os.path.exists(LABELS_PATH):
        st.error(f"Labels file not found: {LABELS_PATH}")
        st.stop()

    try:
        with open(MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)
            model = model_data['model'] if isinstance(model_data, dict) else model_data
        
        with open(LABELS_PATH, 'rb') as f:
            label_encoder = pickle.load(f)
        
        st.success("Model loaded successfully!")
        return model, label_encoder
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model, label_encoder = load_model_and_labels()

# ============================
# Initialize MediaPipe
# ============================
@st.cache_resource
def get_mediapipe_hands():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.7
    )
    return hands, mp_hands, mp_drawing, mp_drawing_styles

hands, mp_hands, mp_drawing, mp_drawing_styles = get_mediapipe_hands()

# ============================
# Process Frame
# ============================
def process_frame(frame):
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    data_aux = [0] * 84
    bbox_coords = None

    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
            x_, y_ = [], []
            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)

            hand_data = []
            for lm in hand_landmarks.landmark:
                hand_data.append(lm.x - min(x_))
                hand_data.append(lm.y - min(y_))

            start_idx = i * 42
            data_aux[start_idx:start_idx + 42] = hand_data

            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Bounding box
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            bbox_coords = (x1, y1)

    return frame, data_aux, bbox_coords

# ============================
# Predict
# ============================
def predict_sign(data_aux):
    if sum(data_aux) == 0:  # No hand detected
        return None
    try:
        features = np.array(data_aux).reshape(1, -1)
        pred = model.predict(features)
        label = label_encoder.inverse_transform(pred)[0]
        return label
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return None

# ============================
# Webcam Mode
# ============================
st.subheader("📹 Live Webcam Detection")

col1, col2 = st.columns(2)
with col1:
    start_btn = st.button("Start Webcam", type="primary")
with col2:
    stop_btn = st.button("Stop Webcam")

FRAME_WINDOW = st.image([])
camera = None

if start_btn:
    st.session_state.running = True
if stop_btn:
    st.session_state.running = False

if getattr(st.session_state, "running", False):
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        st.error("Cannot access camera. Trying uploaded image mode instead.")
        st.session_state.running = False
    else:
        st.info("Camera started! Show your hand signs clearly.")

while getattr(st.session_state, "running", False):
    ret, frame = camera.read()
    if not ret:
        st.error("Failed to capture frame.")
        break

    frame = cv2.flip(frame, 1)  # Mirror effect
    processed_frame, data_aux, bbox = process_frame(frame.copy())
    
    prediction = predict_sign(data_aux)
    if prediction and bbox:
        x1, y1 = bbox
        cv2.putText(processed_frame, str(prediction), (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)

    FRAME_WINDOW.image(processed_frame, channels="BGR")

# Release camera if used
if camera:
    camera.release()

# ============================
# Upload Image Mode
# ============================
st.markdown("---")
st.subheader("🖼️ Or Upload an Image")

uploaded_file = st.file_uploader("Choose an image of a hand sign", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    processed_frame, data_aux, bbox = process_frame(frame.copy())

    prediction = predict_sign(data_aux)
    
    if prediction:
        if bbox:
            x1, y1 = bbox
            cv2.putText(processed_frame, str(prediction), (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
        st.success(f"**Predicted Sign: {prediction}**")
    else:
        st.warning("No hand detected or unable to classify.")

    st.image(processed_frame, channels="BGR", caption="Detection Result")

# ============================
# Footer
# ============================
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, MediaPipe & scikit-learn")