# 🎓 AI Body Language and Emotion Reader
### E-Learning Student Engagement & Attention Tracker

A real-time AI-powered student engagement tracking system that uses webcam input, MediaPipe Holistic landmark extraction, and a Random Forest Classifier to detect and classify student behavioral states — **Attentive**, **Distracted**, or **Drowsy** — and display them live on a Streamlit web dashboard.

---

## 📋 Project Overview

| Item | Details |
|---|---|
| **Student** | 2021WA86338 |
| **Institution** | BITS Pilani, Hyderabad Campus |
| **Program** | MTech Computing Systems & Infrastructure (WILP) |
| **Course** | CSIWZG628T Dissertation |
| **Supervisor** | [Supervisor Name] |
| **Model Accuracy** | 99.62% |

---

## 🗂️ Project Structure

```
BODYLANGUAGEEMOTIONREADER/
│
├── data/
│   └── engagement_coords.csv       # Collected landmark coordinate dataset
│
├── model/
│   ├── engagement_model.pkl        # Trained Random Forest Classifier
│   ├── confusion_matrix.png        # Model evaluation confusion matrix
│   └── model_report.txt            # Accuracy and classification report
│
├── src/  (or root folder)
│   ├── data_collection.py          # Step 1: Webcam data collection script
│   ├── train_model.py              # Step 2: Model training script
│   └── app.py                      # Step 3: Live Streamlit dashboard
│
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- A working webcam (720p or higher recommended)
- Visual Studio Code (recommended IDE)

### Step 1 — Clone or download the project
```bash
git clone https://github.com/yourusername/BODYLANGUAGEEMOTIONREADER.git
cd BODYLANGUAGEEMOTIONREADER
```

### Step 2 — Create a virtual environment (recommended)
```bash
python -m venv engagement_env
```

Activate it:
- **Windows:** `engagement_env\Scripts\activate`
- **Mac/Linux:** `source engagement_env/bin/activate`

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

Run the three scripts **in order**:

### 1️⃣ Collect Training Data
```bash
python data_collection.py
```
- Open the script and set `CURRENT_CLASS` to `'Attentive'`, `'Distracted'`, or `'Drowsy'`
- A webcam window will open with MediaPipe landmarks drawn on your face and body
- Press **Q** to stop recording
- Repeat for all three classes

> Recommended: collect at least **300–500 frames per class** for best accuracy

### 2️⃣ Train the Model
```bash
python train_model.py
```
- Loads `engagement_coords.csv` and trains a Random Forest Classifier
- Prints accuracy score and classification report to terminal
- Saves the trained model as `engagement_model.pkl`
- Saves confusion matrix as `model/confusion_matrix.png`
- Saves full report as `model/model_report.txt`

### 3️⃣ Launch the Live Dashboard
```bash
streamlit run app.py
```
- Opens a browser window automatically
- Shows live webcam feed with MediaPipe landmark overlay
- Displays current engagement state (color-coded) and confidence bar chart in real time

---

## 📊 Model Performance

Trained and evaluated on 1,330 samples (80/20 train-test split):

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Attentive | 0.99 | 1.00 | 0.99 | 95 |
| Distracted | 1.00 | 1.00 | 1.00 | 100 |
| Drowsy | 1.00 | 0.99 | 0.99 | 71 |
| **Overall Accuracy** | | | **99.62%** | **266** |

---

## 🧠 How It Works

```
Webcam Feed
    ↓
MediaPipe Holistic API
(extracts face + body landmarks simultaneously)
    ↓
Coordinate Flattening
(500 landmarks × 4 values = 2,000 features per frame)
    ↓
Random Forest Classifier
(100 decision trees, trained on labeled coordinate data)
    ↓
Streamlit Dashboard
(real-time engagement state + confidence scores)
```

---

## 🛠️ Tech Stack

| Library | Version | Purpose |
|---|---|---|
| OpenCV | 4.13.0.92 | Webcam capture and frame processing |
| MediaPipe | 0.10.33 | Face and body landmark extraction |
| scikit-learn | 1.8.0 | Random Forest model training and evaluation |
| Pandas | 3.0.2 | Dataset handling and feature formatting |
| NumPy | 2.4.4 | Numerical operations |
| Matplotlib | 3.10.8 | Confusion matrix visualization |
| Streamlit | 1.58.0 | Live web dashboard |

---

## 🔮 Future Improvements

- [ ] Collect data from multiple subjects for better generalization
- [ ] Add temporal smoothing (rolling average queue) to stabilize predictions
- [ ] Implement session logging to track engagement trends over time
- [ ] Add audio alert when Drowsy state is detected for extended periods
- [ ] Convert model to TensorFlow Lite for mobile/edge deployment
- [ ] Extend to multi-student monitoring

---

## ⚠️ Known Limitations

- Model trained on single-subject data — accuracy may vary for different individuals
- Performance may degrade under poor lighting or extreme head angles
- Dashboard currently supports single-student monitoring only
- Requires a front-facing webcam with clear view of face and upper body

---

## 📄 License

This project is submitted as part of the CSIWZG628T Dissertation requirement at BITS Pilani, Hyderabad Campus. All rights reserved.
