# Rock-Paper-Scissor: Image Classification
# Image Processing & Data Augmentation Techniques
To enhance model performance and address potential overfitting, the following data augmentation methods were employed:

1. Rescaling: Normalizes pixel values to the range [0, 1], mitigating scale sensitivity.
2. Rotation: Randomly rotates images up to 20 degrees, increasing orientation robustness.
3. Zoom: Allows random zooms up to 5%, enhancing size and scale invariance.
4. Width and Height Shift: Random shifts up to 5% in both axes, boosting position invariance.
5. Shear: Applies shear transformations randomly, aiding in distortion robustness.
6. Horizontal Flip: Flips images horizontally, improving orientation invariance.

These augmentations contribute to model robustness by introducing variability in the training data.

# Model Architecture
Convolutional Neural Network (CNN):

Input: Grayscale images (300x300 pixels).
Layers:
Three convolutional layers (64, 32, 32 filters respectively) with 3x3 kernel sizes.
MaxPooling layers for downsampling.
Dropout layer for overfitting mitigation.
Flatten layer to transform 2D output to 1D.
Two dense layers (64 and 3 neurons).
Activation: ReLU for intermediate layers and softmax for output.
Designed to efficiently capture image features for rock-paper-scissors classification.

# Models & Hyperparameters
Three distinct models were explored:

Model 1: Deep feedforward network with three dense layers (512, 256, 3 neurons).
Performance: 33% accuracy, 2.4 loss.
Model 2: CNN with two convolutional layers.
Performance: 78% accuracy, 0.5 loss.
Model 3 (Chosen Model): Enhanced CNN with three convolutional layers and a 50% dropout.
Performance: 78% accuracy, 0.5 loss.
Model visualizations and detailed results can be found in the provided figures.

# Rock-Paper-Scissor Game
For practical application, a game function was developed:

Inputs: Two images and the model.
Output: Game result based on model predictions.

# Conclusion
The finalized model demonstrates substantial accuracy for both training and validation. While the current results are promising, there's potential for further improvement. This project provided hands-on experience in applying foundational image processing techniques to real-world computer vision challenges, emphasizing the importance of hyperparameter tuning. The game function further showcased the model's real-time applicability.
