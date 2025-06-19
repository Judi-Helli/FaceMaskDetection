# Face Mask Detection using Deep Learning

This project focuses on building and evaluating various deep learning models to detect whether a person is wearing a face mask correctly, incorrectly, or not at all. The best-performing model is then deployed in a Streamlit web application for real-time inference.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Models](#models)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Web Application](#web-application)

## Introduction

The primary goal of this project is to automate the process of face mask detection in public spaces to ensure health and safety compliance. This is accomplished by leveraging deep learning models to classify images of faces into three distinct categories: "With Mask," "Without Mask," and "Mask Worn Incorrectly".

## Dataset

The project utilizes the "Face Mask Detection" dataset from Kaggle, which contains a total of 8,982 images, with 2,994 images for each class. The images have been pre-cropped to focus on the face and head region, and they are resized to 224x224 pixels and normalized before being fed into the models. The dataset is split into training (70%), validation (15%), and testing (15%) sets.

## Models

Four different models were trained and evaluated to find the most effective solution:

1.  **Custom CNN**: A baseline convolutional neural network built from scratch, achieving an accuracy of approximately 96%.
2.  **MobileNetV2**: A transfer learning model that uses MobileNetV2 as a frozen feature extractor. It is fast to train and provides high accuracy.
3.  **ResNet50**: Another transfer learning model that uses a pre-trained ResNet50. This model showed moderate performance and was not selected for deployment.
4.  **DenseNet121**: The best-performing model in this project, which uses a fine-tuned, pre-trained DenseNet121. It achieved the highest overall accuracy and generalization.

## Getting Started

To get started with this project, clone the repository and install the required dependencies.

### Installation

The necessary Python libraries are listed in the `requirements.txt` file:

streamlit
tensorflow
numpy
pillow

You can install them using pip:
pip install -r FaceMaskDetection/face-mask-app/requirements.txt


### Usage
The repository includes scripts for training and evaluating the models.

Training
To train a model, run the corresponding training script from the scripts/ directory. For example, to train the DenseNet121 model, run:

python FaceMaskDetection/scripts/train_densenet121.py


Evaluation
To evaluate a trained model, use the evaluation scripts. For example, to evaluate the DenseNet121 model, run:

python FaceMaskDetection/scripts/evaluate_densenet121.py


### Web Application
This project includes a web application built with Streamlit that allows you to perform face mask detection using the best-performing model (DenseNet121). You can either upload an image or use your webcam to capture a live picture.

Running the App
To run the web application, navigate to the face-mask-app directory and run the following command:

streamlit run FaceMaskDetection/face-mask-app/app.py

The app will open in your web browser, where you can upload an image or use your webcam to get a prediction and a confidence score.
