# 🧵 Fabric Defect Detection using Deep Learning
## 📌 Overview

Fabric Defect Detection is an AI-powered web application that automatically identifies defects in textile fabrics using a deep learning model based on **Transfer Learning (MobileNetV2)**.

The system allows users to upload a fabric image and instantly predicts the defect type, displays the prediction confidence, generates a Grad-CAM heatmap highlighting the defect region, provides quality ratings, maintenance suggestions, and allows downloading a PDF report.

---

## ✨ Features

- 📷 Upload fabric images
- 🤖 AI-based defect classification
- 📊 Prediction confidence score
- 🔥 Grad-CAM heatmap visualization
- ⭐ Fabric quality rating
- 💡 Maintenance suggestions
- 📄 Download PDF report
- 🎨 User-friendly Flask interface

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|----------|
| Python | Programming Language |
| Flask | Backend Framework |
| TensorFlow/Keras | Deep Learning |
| MobileNetV2 | Transfer Learning Model |
| OpenCV | Image Processing |
| NumPy | Numerical Computing |
| Matplotlib | Visualization |
| Pillow | Image Handling |
| HTML/CSS | Frontend |

---

# 📂 Project Structure

```text
Fabric-Defect-Detection/
│
├── app.py
├── train_model.py
├── split_data.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── dataset/
│
├── model/
│   ├── fabric_transfer_model.h5
│   └── class_names.json
│
├── static/
│   ├── uploads/
│   ├── heatmap.png
│   └── styles/
│
├── templates/
│   ├── index.html
│   ├── result.html
│   └── pdf_template.html
│
└── screenshots/
    ├── home.png
    ├── result.png
    ├── chart.png
    └── heatmap.png
```

---

# 🧠 Model Architecture

```
Input Image
      │
      ▼
Image Preprocessing
      │
      ▼
MobileNetV2
(Transfer Learning)
      │
      ▼
Global Average Pooling
      │
      ▼
Dropout
      │
      ▼
Dense Layer
      │
      ▼
Softmax Classifier
      │
      ▼
Prediction
```

---

# 🔄 Workflow

```
Upload Image
      │
      ▼
Image Preprocessing
      │
      ▼
Deep Learning Model
      │
      ▼
Prediction
      │
      ├── Confidence Score
      ├── Grad-CAM Heatmap
      ├── Quality Rating
      ├── Suggestions
      ▼
Display Result
```

---

# 📊 Dataset

The model is trained on textile fabric images classified into the following categories:

- Good Fabric
- Horizontal Defect
- Vertical Defect
- Hole
- Line Defect
- Stain

The dataset is automatically split into training and testing sets before training.

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/srinivasprasad10/Fabric-Defect-Detection.git
```

Move into the project

```bash
cd Fabric-Defect-Detection
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Train the Model

```bash
python split_data.py
python train_model.py
```

---

# ▶️ Run the Application

```bash
python app.py
```

Open your browser and visit:

```
http://127.0.0.1:5000
```

---

# 📸 Application Screenshots

## Home Page

![Home](screenshots/home.png)

---

## Prediction Result

![Prediction](screenshots/result.png)

---

## Confidence Chart

![Chart](screenshots/chart.png)

---

## Grad-CAM Heatmap

![Heatmap](screenshots/heatmap.png)

---

# 📈 Sample Output

- ✔ Predicted Defect
- ✔ Confidence Score
- ✔ Defect Explanation
- ✔ Fabric Quality Rating
- ✔ Suggestions
- ✔ Grad-CAM Visualization
- ✔ PDF Report Generation

---

# 🎯 Future Enhancements

- Live Camera Detection
- YOLOv8-based Object Detection
- Cloud Deployment
- REST API Support
- Multi-language Support
- Real-time Manufacturing Integration

---

# 👨‍💻 Author

**Srinivas Prasad**

GitHub: https://github.com/srinivasprasad10
