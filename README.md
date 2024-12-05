# Kitchenware Classification Project

## Overview
This project demonstrates the use of deep learning for the classification of kitchenware items (e.g., scissors, cups, knives, spoons). It includes custom-designed models (ANN and CNN) as well as transfer learning using VGG16. Additionally, it features a real-time object detection system using OpenCV.

The primary focus of this project is:
- Preprocessing and augmenting image data.
- Training models to classify kitchenware items.
- Deploying real-time detection using a webcam.

## Features
- **Custom Models**: Both Artificial Neural Network (ANN) and Convolutional Neural Network (CNN) architectures.
- **Transfer Learning**: Utilization of VGG16 pre-trained on ImageNet for enhanced performance.
- **Data Augmentation**: Includes rotation, scaling, flipping, and zooming for improved generalization.
- **Real-Time Detection**: Real-time object classification using a trained model and OpenCV.

---

## File Structure
The project is organized as follows:
kitchenware-classification/ ├── README.md # Project overview and instructions ├── requirements.txt # List of Python dependencies ├── src/ # Source code for models and utilities │ 
├── data_preprocessing.py # Preprocessing and data augmentation │ ├── cnn_model.py # CNN model (includes callbacks and training logic) │ ├── ann_model.py # ANN model definition and callbacks │ 
├── transfer_learning.py # VGG16-based transfer learning model │ ├── real_time_detection.py # Real-time detection with OpenCV ├── examples/ # Example scripts for training and evaluation │ 
├── train_ann_with_callbacks.py # Training script for ANN │ ├── train_vgg16_with_callbacks.py # Training script for VGG16 │ ├── evaluate_model.py # Evaluation script for trained models ├── datasets/ # Placeholder for datasets │ 
├── train/ # Training images (organized into subfolders by class) │ ├── validation/ # Validation images (organized into subfolders by class) │ ├── test/ # Testing images (organized into subfolders by class)

## Installation
### Prerequisites
- Python 3.7 or higher
- Virtual environment (recommended)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/kitchenware-classification.git
   cd kitchenware-classification
   
Install dependencies:
pip install -r requirements.txt

The dataset should be organized as follows:
datasets/
├── train/
│   ├── scissors/
│   ├── cups/
│   ├── knives/
│   ├── spoons/
├── validation/
│   ├── scissors/
│   ├── cups/
│   ├── knives/
│   ├── spoons/
├── test/
│   ├── scissors/
│   ├── cups/
│   ├── knives/
│   ├── spoons/

Training a Model:
python /train_ann_with_callbacks.py
python /cnn_model.py
python /train_vgg16_with_callbacks.py

Evaluate a trained model using the test dataset:
python examples/evaluate_model.py

Run the real-time detection script with a webcam:
python src/real_time_detection.py







