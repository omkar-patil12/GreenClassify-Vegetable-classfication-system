# GreenClassify-Vegetable-classfication-system
# 📌 Project Summary

VegiVision is an AI-powered image classification system designed to recognize and categorize vegetables from uploaded images.
The project leverages Transfer Learning with MobileNetV2 to achieve high accuracy while maintaining efficient training time.

The trained deep learning model is saved in .h5 format and deployed using a Flask-based web application for real-time prediction.

## 🧠 Model Architecture

This project uses a pretrained convolutional neural network as the feature extractor.

Base Model: MobileNetV2 (Pretrained on ImageNet)

Approach: Transfer Learning

Custom Classification Layers:

GlobalAveragePooling2D

Dropout (0.3)

Dense Layer (128 units, ReLU activation)

Output Layer (15 classes, Softmax activation)

This approach allows faster convergence and improved performance with limited training time.

## 📂 Dataset Information

The dataset consists of labeled vegetable images organized into class-wise folders.

- Total Classes: 15

- Images per Class: Approximately 1,000

- Total Images: ~45,000

- Image Format: JPG/PNG

- Input Size: 224 × 224 pixels

## 🚀 Training Configuration
Parameter	Value
Image Size	224 × 224
Batch Size	32
Optimizer	Adam
Loss Function	Categorical Crossentropy
Epochs	With Early Stopping
Final Test Accuracy	~99%
📊 Data Augmentation Techniques

To improve generalization and prevent overfitting:

Random Rotation

Zoom Range

Horizontal Flip

Rescaling (1/255 normalization)

EarlyStopping and ModelCheckpoint callbacks were used to store the best performing model.

## 🌐 Web Application Workflow

The trained model is integrated into a Flask web application for live predictions.

### 🔄 Prediction Pipeline

User uploads an image

Image resized to 224×224

Pixel values normalized (1/255)

Model predicts vegetable category

Output displayed with confidence score

## 🖥️ Frontend Structure
### 1️⃣ Home Page (index.html)

Displays project title

“Start Prediction” button

Redirects to prediction page

### 2️⃣ Prediction Page (predict.html)

Image upload field

Submit button

Sends image to Flask backend

### 3️⃣ Result Page (result.html)

Shows:

Predicted Vegetable Name

Confidence Percentage

Options:

Predict Another

Back to Home

## 🛠️ Installation Guide
### 1️⃣ Clone Repository
git clone https://github.com/omkar-patil12/GreenClassify-Vegetable-classfication-system
cd vegivision

### 2️⃣ Install Required Packages
pip install tensorflow flask numpy pillow

### 3️⃣ Train the Model (Optional)
python train.py

### 4️⃣ Run the Web Application
python app.py


#### Open in browser:

http://127.0.0.1:5000/

## 📈 Key Features

 - High Accuracy (~99%)
 - Transfer Learning Implementation
 - Real-time Image Prediction
 - User-Friendly Web Interface
 - Optimized Model Deployment

## 🎯 Skills Demonstrated

Deep Learning with TensorFlow/Keras

Convolutional Neural Networks (CNNs)

Transfer Learning (MobileNetV2)

Image Preprocessing & Augmentation

Model Evaluation & Optimization

Flask Web Development

Model Deployment


### 👨‍💻 Developer

Omkar Ram Patil <br>
B.Tech Computer Science ( AIML ) <br>
D Y Patil Agricultural & Technical Unversity,Talsande
