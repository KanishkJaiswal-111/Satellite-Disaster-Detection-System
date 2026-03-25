 🌍 AI-Based Satellite Disaster Detection System

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-FF6F00?style=for-the-badge&logo=tensorflow)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-FF4B4B?style=for-the-badge&logo=streamlit)

## 📌 Overview
This project implements an end-to-end Deep Learning computer vision pipeline to classify high-resolution satellite imagery into four distinct disaster topologies:
1. **Damage** (Post-hurricane/earthquake structural ruin)
2. **Flood** (Inundated zones)
3. **Wildfire** (Active burns and scorched earth)
4. **Normal** (Baseline terrain)

By leveraging **EfficientNetB4** via transfer learning, the model achieves a **96% global F1-Score**. A custom **Grad-CAM** visual explainability module provides complete transparency by overlaying the structural features the neural network uses to make its predictions. 

## ✨ Key Features
- **State-of-the-Art Architecture:** Utilizes `EfficientNetB4` trained with a two-stage gradient freeze/unfreeze methodology.
- **Handling Extreme Imbalance:** Overcame heavy minority class imbalances using strategic subset oversampling and dynamic spatial data augmentation (rotation, zoom, brightness, multi-axis flipping).
- **Explainable AI (XAI):** Integrated OpenCV-powered Gradient-weighted Class Activation Mapping (Grad-CAM) to draw heatmap attention masks over satellite topographies.
- **Real-Time Deployment:** Deployed a fully functional, interactive frontend web application using **Streamlit**.

## 🚀 Installation & Usage
### 1. Clone the repository
```bash
git clone https://github.com/Kanishk.Jaiswal-111/satellite-disaster-detection.git
cd satellite-disaster-detection
```

### 2. Install Dependencies
Make sure you have Python 3.12+ and the completely up-to-date Keras 3 installed.
```bash
pip install -r requirements.txt
```
*(This will install `tensorflow`, `opencv-python`, `streamlit`, `Pillow`, `numpy`, and `scikit-learn`)*

### 3. Download the Model
Because the custom `EfficientNetB4` weights (`model.keras`) exceed GitHub's 100MB file limit, they are not tracked in this repository. 
- Download the `model.keras` file from [Google Drive Link Here](#)
- Place the downloaded `model.keras` file directly inside the `App/` folder.

### 4. Run the Streamlit Application
```bash
cd App
streamlit run app.py
```

## 🧠 Model Training Details
The model was trained using `ImageDataGenerator` with categorical crossentropy loss. The optimization pipeline employed:
- **Optimizer:** Adam (`learning_rate=1e-5` for microscopic fine-tuning)
- **EarlyStopping:** Terminated training proactively when `val_loss` plateaued to absolutely prevent overfitting.
- **ReduceLROnPlateau:** Dynamically scaled learning rates during boundary convergence.

## 📁 Repository Structure
```text
├── App/                   # Web application deployment directory
│   ├── app.py             # Streamlit frontend & prediction script
│   └── model.keras        # Trained Neural Network weights (Download separately)
├── data/                  # 4-class satellite image dataset
├── src/                   # Jupyter notebooks and backend scripts
│   ├── main.ipynb         # Full data augmentation & 2-stage training pipeline
│   └── gradcam.py         # Heatmap generation and OpenCV overlay methodology
├── .gitignore             # Ignores large binary artifacts (*.keras, *.h5)
└── README.md              # You are here!
```
