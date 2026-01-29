# CIFAR-100 Image Classification — ANN vs CNN
### 📌 Project Overview# Image_classifiaction
This project focuses on multi-class image classification using the CIFAR-100 dataset. The primary objective is to compare the performance of Artificial Neural Networks (ANN) and Convolutional Neural Networks (CNN) on a challenging 100-class RGB image dataset.

The project demonstrates why CNNs are more suitable than traditional fully connected networks for computer vision tasks and includes model training, evaluation, and visualization of correct and incorrect predictions.

## 🎯 Objectives

- Perform 100-class image classification using CIFAR-100

- Compare ANN and CNN architectures on image data

- Evaluate model performance using accuracy and loss metrics

- Visualize correct and incorrect predictions

- Analyze model errors across fine-grained object categories

- Demonstrate best practices in deep learning for computer vision

## 📦 Dataset
- CIFAR-100

- Total Images: 60,000

- Training Images: 50,000

- Test Images: 10,000

- Image Size: 32 × 32 pixels

- Channels: 3 (RGB)

- Number of Classes: 100 (fine labels)

- Superclasses: 20 (coarse labels)

The dataset contains diverse object categories including animals, vehicles, household objects, plants, and people.

### Official Source:
https://www.cs.toronto.edu/~kriz/cifar.html

## 🧠 Models Implemented
### 1. Artificial Neural Network (ANN)

- Input: Flattened 32×32×3 images

- Fully connected dense layers

- ReLU activations

- Softmax output layer (100 classes)

### Purpose:
To establish a baseline and demonstrate limitations of fully connected networks for image data.

### 2. Convolutional Neural Network (CNN)

- Convolutional layers (Conv2D)

- MaxPooling layers

- Batch Normalization (optional)

- Dropout for regularization

- Fully connected classifier head

- Softmax output layer (100 classes)

### Purpose:
To leverage spatial feature learning for improved image classification performance.

## 🔄 Data Preprocessing

- Normalize pixel values to range [0, 1]

- One-hot encode labels for softmax classification

- Flatten labels for evaluation and visualization

- Shuffle and batch data for efficient training

Example:

x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32') / 255.0

y_train_one_hot = to_categorical(y_train, 100)
y_test_one_hot  = to_categorical(y_test, 100)

## 🏗️ Project Structure

cifar100-ann-vs-cnn/
│
├── data/
│   └── (optional local dataset files)
│
├── notebooks/
│   ├── cifar100_data_exploration.ipynb
│   ├── ann_model_training.ipynb
│   ├── cnn_model_training.ipynb
│   └── visualization_and_error_analysis.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── ann_model.py
│   ├── cnn_model.py
│   └── evaluation.py
│
├── results/
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── prediction_visualizations.png
│
├── requirements.txt
├── README.md
└── LICENSE

## 📊 Evaluation Metrics

- Accuracy

- Training vs Validation Loss

- Training vs Validation Accuracy

- Visual inspection of correct and incorrect predictions

- Confusion matrix (optional for deeper analysis)

## 🏆 Results Summary (Example — Replace with Your Actual Numbers)

Model	Test Accuracy	Observations

ANN	~20% – 30%	Struggles due to loss of spatial information
CNN	~45% – 65%+	Learns spatial features, significantly better performance

## 🔍 Key Insights

- CNN significantly outperforms ANN on CIFAR-100 due to spatial feature learning.

- ANN fails to capture local patterns and spatial hierarchies in images.

- CNN learns hierarchical features such as edges, textures, and object parts.

- Misclassifications commonly occur between visually similar object categories.

- Model architecture plays a critical role in image classification performance.

## 🧾 Conclusion

- CNN is the preferred architecture for CIFAR-100 image classification.

- ANN provides a useful baseline but is not suitable for complex image data.

- The project demonstrates the importance of convolutional layers for computer vision tasks.

- This work reinforces best practices for deep learning on image datasets.

## 🖼️ Prediction Visualization

The project includes visualization of correctly and incorrectly classified images, highlighting:

Correct predictions (green labels)

Incorrect predictions (red labels with predicted vs true class)

Model strengths and common failure cases
