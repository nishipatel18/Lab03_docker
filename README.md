# Docker ML Labs 🐳

## Overview
This repository contains two Docker-based Machine Learning labs. Each lab trains a different ML model on a different dataset and serves predictions via a web interface.

---

## Lab 1 - Wine Classification

### Dataset
- **Name:** Wine Dataset (sklearn)
- **Samples:** 178
- **Features:** 13 (alcohol, malic acid, ash, etc.)
- **Classes:** 3 wine categories

### Model
- **Algorithm:** Logistic Regression
- **Library:** scikit-learn
- **Test Size:** 20%
- **Accuracy:** ~100%

### How It Works
The model is trained on the Wine dataset and saved as `wine_model.pkl`. The container runs the training script and prints the accuracy.

### File Structure
```
Lab1/
├── src/
│   ├── main.py
│   └── requirements.txt
├── dockerfile
└── ReadMe.md
```

### How to Run
```bash
cd Lab1
docker build -t lab1:v1 .
docker run lab1:v1
```

### Result
```
Model Accuracy: 100.00%
The model training was successful
```

---

## Lab 2 - Breast Cancer Classification

### Dataset
- **Name:** Breast Cancer Dataset (sklearn)
- **Samples:** 569
- **Features:** 30 (radius, texture, perimeter, area, etc.)
- **Classes:** 2 (Benign, Malignant)

### Model
- **Algorithm:** Deep Neural Network (TensorFlow/Keras)
- **Layers:** 4 (64 → 32 → 16 → 1)
- **Activation:** ReLU + Sigmoid
- **Loss:** Binary Crossentropy
- **Epochs:** 50

### How It Works
- **Service 1 (model-training):** Trains the neural network and saves it as `my_model.keras`
- **Service 2 (serving):** Loads the saved model and serves a Flask web app on port 4000

### File Structure
```
Lab2/
├── src/
│   ├── templates/
│   │   └── predict.html
│   ├── statics/
│   ├── main.py
│   └── model_training.py
├── dockerfile
├── docker-compose.yml
├── requirements.txt
└── HOWTO
```

### How to Run
```bash
cd Lab2
docker-compose up --build
```

Then open your browser at:
```
http://localhost/predict
```

### Result
Enter the 30 feature values and click **Analyze** to get:
- 🟢 **BENIGN** — non-cancerous
- 🔴 **MALIGNANT** — cancerous

---

## Technologies Used
- Python 3.10
- Docker & Docker Compose
- scikit-learn
- TensorFlow / Keras
- Flask
