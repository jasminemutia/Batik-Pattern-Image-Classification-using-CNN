# Batik Pattern Classification Using Convolutional Neural Network (CNN)
This project focuses on classifying traditional Indonesian batik patterns using a Convolutional Neural Network (CNN). 
It includes five distinct classes of batik: batik-sekar, batik-sidoluhur, batik-sidomukti, batik-sogan, and batik-tambal. 
This repository demonstrates the data preparation, model architecture, and evaluation process to build a CNN capable of recognizing these patterns.

## Data Preparation
The batik images were loaded, resized, and converted into arrays. We created one-hot encoded labels for each class and stored the image data and labels into arrays for model training. 
A histogram color analysis was performed to examine the color distribution in the images.

## Data Augmentation
To increase the training data variety and reduce overfitting, we applied data augmentation techniques, including zoom, shear, and vertical flips.

## Dataset Splitting
The dataset was split into 80% training, 10% validation, and 10% testing sets to evaluate the model effectively.

## Model Architecture
A CNN model with five convolutional layers was designed. Each layer uses varying kernel sizes and incorporates ReLU activation and MaxPooling layers to capture complex patterns in the batik images. 
The model ends with fully connected (dense) layers before the output layer.

## Training and Evaluation
The model was trained for 30 epochs with categorical_crossentropy as the loss function and adam optimizer. 
Despite tuning, the model reached a modest accuracy of 30% on the test set, highlighting areas for further improvement.

## Classification Metrics
The final results include a classification report with precision, recall, and F1-score for each batik class, comparing the ground truth and predicted labels.

## Results
The model achieved a 30% accuracy on the test set. A classification report indicated variable performance across classes, with some batik patterns classified more accurately than others.

## Future Improvements
- Hyperparameter Tuning: Further tuning may improve model accuracy.
- Model Architecture: Experimenting with alternative architectures or pre-trained models could enhance performance.
- Data Augmentation: Increasing augmentation techniques could improve generalization.
