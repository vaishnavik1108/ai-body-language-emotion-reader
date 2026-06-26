import cv2
import mediapipe as mp
import csv
import os

# =======================================================
# 1. CHOOSE YOUR CLASS LABEL HERE BEFORE RUNNING!
# Options: 'Attentive', 'Distracted', 'Drowsy'
CURRENT_CLASS = 'Attentive'

# 2. SET MAX FRAMES TO COLLECT PER SESSION (prevents runaway recording)
MAX_FRAMES = 500
# =======================================================

# Save to data/ folder — matches train_model.py path
csv_filename = os.path.join('data', 'engagement_coords.csv')

# Ensure the data/ folder exists
os.makedirs('data', exist_ok=True)

# MediaPipe configurations
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Fixed number of landmarks per frame for consistent CSV columns
FIXED_LANDMARKS = 500

# Initialize CSV with column headers if it doesn't exist yet
if not os.path.exists(csv_filename):
    headers = ['class']
    for val in range(1, FIXED_LANDMARKS + 1):
        headers += [f'x{val}', f'y{val}', f'z{val}', f'v{val}']
    with open(csv_filename, mode='w', newline='') as f:
        csv.writer(f).writerow(headers)
    print(f"📁 Created new dataset file: {csv_filename}")
else:
    print(f"📁 Appending to existing dataset: {csv_filename}")

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ ERROR: Could not open webcam. Check your camera connection.")
    exit()

print(f"🔴 Starting data collection for class: [{CURRENT_CLASS}]")
print(f"📌 Will collect up to {MAX_FRAMES} frames. Press Q to stop early.")
print("👉 Click on the popup camera window so your computer focuses on it!")

record_count = 0

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Stop automatically once MAX_FRAMES is reached
        if record_count >= MAX_FRAMES:
            print(f"✅ Reached MAX_FRAMES limit of {MAX_FRAMES}. Stopping.")
            break

        ret, frame = cap.read()
        if not ret:
            print("⚠️ Warning: Failed to read frame from webcam.")
            break

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)

        # Draw visible meshes on screen
        if results.face_landmarks:
            mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        # Extract pose and face landmarks into standard Python lists
        pose = list(results.pose_landmarks.landmark) if results.pose_landmarks else []
        face = list(results.face_landmarks.landmark) if results.face_landmarks else []
        all_landmarks = pose + face

        if len(all_landmarks) > 0:
            row = []
            for landmark in all_landmarks[:FIXED_LANDMARKS]:
                row.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])

            # Pad with zeros if fewer than FIXED_LANDMARKS were detected
            while len(row) < (FIXED_LANDMARKS * 4):
                row.extend([0.0, 0.0, 0.0, 0.0])

            # Prepend class label and save to CSV
            row.insert(0, CURRENT_CLASS)
            with open(csv_filename, mode='a', newline='') as f:
                csv.writer(f).writerow(row)

            record_count += 1

        # Show progress on the webcam window
        progress = f"{record_count}/{MAX_FRAMES}"
        cv2.putText(frame, f"Recording: {CURRENT_CLASS}", (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Saved Frames: {progress}", (15, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, "Press Q to stop early", (15, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

        cv2.imshow('AI Data Collector Dashboard', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            print("⏹️  Recording stopped early by user.")
            break

cap.release()
cv2.destroyAllWindows()
print(f"✅ Successfully saved {record_count} frames for class [{CURRENT_CLASS}] → {csv_filename}")