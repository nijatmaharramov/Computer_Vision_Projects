# Computer-Vision-Projects

# 🥭 Fruits and Vegetables Image Recognition with Deep Learning

This project is a deep learning-based image classifier that identifies fruits and vegetables using TensorFlow and Convolutional Neural Networks (CNN), with additional experiments using transfer learning via ResNet50.

## 📂 Dataset

- **Source**: [Kaggle - Fruit and Vegetable Image Recognition](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition)
- The dataset is organized into `train/`, `test/`, and `validation/` folders with subdirectories representing each class label.

## 🚀 Features

- Custom CNN model with convolutional and dense layers
- Transfer learning using ResNet50 pretrained on ImageNet
- Image preprocessing and visualization
- Training with early stopping and learning rate scheduling
- Single image prediction with probability bar charts

## 📦 Installation

To run the project in Google Colab:

```bash
!pip install kaggle
```

Upload your `kaggle.json` API token file and run:

```python
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d kritikseth/fruit-and-vegetable-image-recognition
!unzip fruit-and-vegetable-image-recognition.zip
```

## 🧠 Model Architectures

### 1. Custom CNN
```python
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224,224,3)),
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(64, 7, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    ...
])
```

### 2. Transfer Learning with ResNet50
```python
base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3), pooling='avg')
base_model.trainable = False

model2 = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Dense(128, activation='relu'),
    ...
])
```

## 🏋️ Training

```python
model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

model.fit(train_data, validation_data=validation_data, epochs=10)
```

Includes:
- Model Checkpointing
- Early Stopping
- Learning Rate Scheduling

## 📊 Evaluation and Prediction

Use the `plot_and_pred()` function to visualize predictions:

```python
plot_and_pred(model, 'path/to/your/image.jpg')
```

This shows:
- The input image
- Predicted class name
- Probability distribution bar chart

## 📈 Sample Results

| Image | Prediction |
|-------|------------|
| ![apple](sample_images/red-apple.jpg) | 🍎 Apple |
| ![watermelon](sample_images/watermelon.jpg) | 🍉 Watermelon |

## 📁 Folder Structure

```
/content/
    ├── train/
    ├── test/
    ├── validation/
    ├── red-apple-freshleaf-dubai-uae-img01.jpg
    └── Watermelon-cuts-24-1.jpg
```

## 🛠 Requirements

- TensorFlow
- NumPy
- Pandas
- Matplotlib
- Kaggle API

Install with:

```bash
pip install tensorflow numpy pandas matplotlib kaggle
```

