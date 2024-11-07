# Batik Pattern Classification Using Convolutional Neural Network (CNN)
This project implements a Convolutional Neural Network (CNN) model to classify five traditional Indonesian batik patterns: batik-sekar, batik-sidoluhur, batik-sidomukti, batik-sogan, and batik-tambal. The primary goal of this project is to leverage CNN to recognize intricate batik patterns from images, offering insights that could benefit cultural studies or be applied within the fashion industry.

## Access the Project
[Batik Pattern Classification Using Convolutional Neural Network (CNN)](https://colab.research.google.com/drive/1GQ3bcawrfKVjdynhN4QwTP6qddoW8o8o?usp=sharing)

## Dataset
The dataset comprises 243 images categorized into five classes representing different batik motifs:
- batik-sekar: 47 images
- batik-sidoluhur: 50 images
- batik-sidomukti: 46 images
- batik-sogan: 50 images
- batik-tambal: 50 images

To streamline the model’s input processing, all images are resized to 64x64 pixels. The dataset is split into 80% for training and 20% for testing, with the testing portion further divided to allocate 10% for validation and 10% for actual testing.

## Data Preparation
The batik images were loaded, resized, and converted into arrays. We created one-hot encoded labels for each class and stored the image data and labels into arrays for model training. 
A histogram color analysis was performed to examine the color distribution in the images.

## Data Augmentation
To improve the model’s ability to recognize patterns under various orientations and scales, data augmentation techniques are applied on the training dataset:
- Shear Range: 0.2
- Zoom Range: 0.2
- Vertical Flip
  
These augmentations help in enhancing the robustness of the model to pattern variations.

## Dataset Splitting
The dataset was split into 80% training, 10% validation, and 10% testing sets to evaluate the model effectively.

## CNN Model Architecture
The CNN model architecture used in this project includes five convolutional layers with varying kernel sizes, pooling layers to reduce dimensionality, and two fully connected layers before the final output layer. The architecture is as follows:
1. Conv2D layer: 64 filters, kernel size 5x5
2. MaxPooling2D layer: pool size 13x13
3. Conv2D layer: 256 filters, kernel size 5x5
4. MaxPooling2D layer: pool size 2x2
5. Conv2D layer: 384 filters, kernel size 3x3
6. Conv2D layer: 384 filters, kernel size 3x3
7. Conv2D layer: 192 filters, kernel size 3x3
8. MaxPooling2D layer: pool size 1x1
9. Flatten layer
10. Fully Connected layer: 4096 units
11. Fully Connected layer: 4096 units
12. Output layer: Softmax activation with 5 outputs, one for each batik class
    
The model has a total of 133,343,173 parameters and is optimized using the categorical cross-entropy loss function and the Adam optimizer.

## Experimental Results
The model was trained for 30 epochs. Despite this training, the accuracy remained relatively low:
- Training Accuracy: 20.6%
- Validation Accuracy: 20.0%
- Test Accuracy: 29.17%
Post-training evaluation through classification_report indicated further tuning might be needed, as seen in the per-class performance:

## Classification Metrics
The final results include a classification report with precision, recall, and F1-score for each batik class, comparing the ground truth and predicted labels.

## Future Improvements
This project highlights the challenges in batik motif classification with CNN, especially with a limited dataset. Current accuracy levels fall short of practical application standards. Potential improvements include:
- Increasing dataset size for better motif pattern recognition.
- Hyperparameter Tuning: Further tuning may improve model accuracy.
- Model Architecture: Experimenting with alternative architectures or pre-trained models could enhance performance. Try transfer learning with a pre-trained model (e.g., ResNet or VGG) for potentially better results with limited data.
- Data Augmentation: Increasing augmentation techniques could improve generalization.

 
