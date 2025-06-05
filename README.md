# Computer-Vision-Projects


# 🧠 Image Classification Projects (Fruits & Emotions)

This repository contains two image classification projects using TensorFlow and CNN architectures, focusing on real-world datasets:

---

## 1. 🍎 Fruit and Vegetable Recognition

A classification project using a custom CNN and transfer learning to recognize different types of fruits and vegetables.

### 📦 Dataset
- [Fruit and Vegetable Image Recognition](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition)

### 🚀 Workflow
- Dataset loading and directory organization
- Custom CNN training
- Transfer learning using ResNet50
- Visualization of predictions
- Accuracy and F1-score tracking

### 🧠 Architectures
- Custom CNN
- ResNet50 (pre-trained on ImageNet)

---

## 2. 🙂 Emotion Recognition from Images

A deep learning model to classify human facial emotions using multiple CNN architectures.

### 📦 Dataset
- [Emotion Recognition Dataset](https://www.kaggle.com/datasets/sujaykapadnis/emotion-recognition-dataset)

### 🚀 Workflow
- Dataset cleanup (removing unwanted categories like "Ahegao")
- Custom CNN model training
- Transfer learning using:
  - ResNet50
  - Xception
- Learning rate scheduling and model checkpointing
- Visual analysis of predictions

### 🧠 Architectures
- Custom CNN
- ResNet50 (with frozen base layers)
- Xception (with frozen base layers)

---

## 📁 Repository Structure

```
.
├── fruit-recognition/
│   ├── Fruit_Detection.ipynb
│
├── emotion-recognition/
│   ├── Facial_emotion_recognition.ipynb
│
└── README.md
```

---

## 📌 Requirements

- Python 3.x
- TensorFlow
- NumPy
- Matplotlib
- Kaggle API (`kaggle.json`)

---

## ▶️ How to Run

1. Upload `kaggle.json` to your Colab environment.
2. Run the respective notebook (`*_setup.ipynb`) to download and prepare the dataset.
3. Train the model using the provided CNN, ResNet50, or Xception architectures.
4. Use `plot_and_pred()` to test your model on unseen images.

---




