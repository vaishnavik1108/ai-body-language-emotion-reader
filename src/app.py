import warnings
import pickle
import os
from collections import deque
from statistics import mode

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st

# ── Suppress noisy library warnings ──
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# ── Page Configuration ──
st.set_page_config(page_title="AI Body Language Reader", layout="wide")
st.title("🎓 AI Body Language and Emotion Reader")
st.subheader("E-Learning Student Engagement Tracker")
st.markdown("---")

# ── Load Model ──
model_path = os.path.join('model', 'engagement_model.pkl')
if not os.path.exists(model_path):
    st.error("❌ Model file not found. Please run `train_model.py` first.")
    st.stop()

with open(model_path, 'rb') as f:
    model = pickle.load(f)

# ── Layout ──
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("📷 Live Stream Tracking")
    frame_placeholder = st.empty()

with col2:
    st.subheader("📊 Live Engagement Metrics")
    chart_placeholder = st.empty()
    st.markdown("---")
    state_placeholder = st.empty()

# ── MediaPipe Setup ──
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
FIXED_LANDMARKS = 500

# ── Rolling Average Queue (smooths flickering predictions) ──
# Keeps last 10 predictions and shows the most common one
SMOOTHING_WINDOW = 10
prediction_queue = deque(maxlen=SMOOTHING_WINDOW)

# ── Webcam ──
#cap = cv2.VideoCapture(0)

@st.cache_resource
def load_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW is faster on Windows
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)     # lower resolution = faster load
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap

cap = load_camera()

if not cap.isOpened():
    st.error("❌ Could not open webcam. Please check your camera connection.")
    st.stop()

# ── Main Loop ──
with mp_holistic.Holistic(min_detection_confidence=0.5,
                           min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Lost webcam connection.")
            break

        # Flip for mirror view and convert to RGB
        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)

        # Draw MediaPipe landmarks on the frame
        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                image_rgb, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image_rgb, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        try:
            # Extract landmarks
            pose = list(results.pose_landmarks.landmark) if results.pose_landmarks else []
            face = list(results.face_landmarks.landmark) if results.face_landmarks else []
            all_landmarks = pose + face

            # Build fixed-length feature vector
            row = []
            for landmark in all_landmarks[:FIXED_LANDMARKS]:
                row.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])

            # Pad with zeros if short
            while len(row) < (FIXED_LANDMARKS * 4):
                row.extend([0.0, 0.0, 0.0, 0.0])

            # Run prediction
            X_live = pd.DataFrame([row])
            raw_class = model.predict(X_live)[0]
            probabilities = model.predict_proba(X_live)[0]

            # Smooth prediction using rolling mode
            prediction_queue.append(raw_class)
            body_language_class = mode(prediction_queue)

            # Color-code the state
            color_map = {
                'Attentive':  '#2ecc71',   # Green
                'Distracted': '#f1c40f',   # Yellow
                'Drowsy':     '#e74c3c',   # Red
            }
            color = color_map.get(body_language_class, '#ffffff')

            # ── Render UI ──
            frame_placeholder.image(image_rgb, channels="RGB",
                                    width="stretch")

            state_placeholder.markdown(
                f"<h2 style='color:{color};'>● CURRENT STATE: "
                f"{body_language_class.upper()}</h2>",
                unsafe_allow_html=True
            )

            chart_data = pd.DataFrame({
                'State': model.classes_,
                'Confidence (%)': probabilities * 100
            })
            chart_placeholder.bar_chart(
                data=chart_data, x='State', y='Confidence (%)', height=280)

        except Exception:
            # If landmark extraction fails, still show the video frame
            frame_placeholder.image(image_rgb, channels="RGB",
                                    width="stretch")

cap.release()
