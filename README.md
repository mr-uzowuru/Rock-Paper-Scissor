# Rock-Paper-Scissor







**A brief description of the selected image processing and data augmentation methods**

1. Rescaling: This method rescales the pixel values of the images by a factor of 1/255 to normalize them to the range [0, 1].It helps prevent the model from becoming too sensitive to the scale of pixel values, making it easier for the model to learn features invariant to changes in brightness and contrast.

2. Rotation Range: This is set to 20 to rotate the input images up to 20 degrees randomly.The method randomly rotates the images by certain degrees within a given range, and it can help the model become more robust to variations in the orientation of the input images.

3. Zoom Range: The room range is set to 0.05 to randomly zoom in or out of the input images up to 5%.The method is mainly used to implement random zooms into or out of the image.It helps to increase the variability of the training data, allowing the model to learn features that are invariant to changes in the size and scale of the images.

4. Width Shift Range and Height Shift Range: Width Shift Range and Height Shift Range are set to 0.05 to randomly shift the input images horizontally and vertically up to 5%.This helps to increase the variability of the training data, allowing the model to learn features that are invariant to changes in the position and location of objects in the images.

5. Shear Range: This method randomly applies a shear transformation to the images by a certain angle within a given range, in this case, 5%.This helps to increase the variability of the training data, allowing the model to become more robust to distortions in the input images.

6. Horizontal Flip: This hyperparameter is set to True to flip the input images horizontally randomly.This helps to increase the variability of the training data, allowing the model to learn features that are invariant to changes in the orientation of objects in the images.


The methods used above increased the training data's variability, making the model more robust to changes in the input images. By applying these methods, we can create a more diverse and extensive training dataset, which helps the model learn more robust features and generalize better to unseen data.


**Final Model Architecture:**

The model architecture used is a simple convolutional neural network (CNN) with three convolutional layers, max-pooling layers, dropout, and dense layers. The model has a total of 5 layers, of which 3 are convolutional layers, 1 dropout layer, and 2 dense layers. The activation function used is ReLU in all layers except the output layer, where the softmax function is used for multi-class classification. 

The first convolutional layer has 64 filters, kernel size of 3x3, and input shape of 300x300x1 (since the input images are grayscale and resized to 300x300). The MaxPooling layer follows this convolutional layer to downsample the output. The next two convolutional layers follow the same pattern, each with 32 filters and a kernel size of 3x3. The MaxPooling layer again downsamples the output.

After the convolutional and pooling layers, a Dropout layer is added to reduce overfitting. This is followed by a Flatten layer to convert the 2D output of the convolutional layers to a 1D vector. Finally, there are two dense layers with 64 and 3 neurons, respectively. The last dense layer has a softmax activation function to output the probabilities of the input image belonging to one of the rock, paper, or scissors classes.

Suitability of the Model and Justification
The given model has enough layers and filters to capture essential features from the input images. It also includes Dropout layers to prevent overfitting; the number of layers is kept minimal to prevent overfitting and improve training speed; the filters in the convolutional layers are designed to extract features from the images that are relevant for the classification task, while the dense layers perform the classification task, The use of max-pooling layers helps to reduce the dimensionality of the feature maps, making the model more efficient.
All this makes this model architecture simple, efficient and suitable for solving the task of classifying rock-paper-scissors images.

The ImageDataGenerator function helps create an augmented dataset that improves the model's accuracy by increasing the diversity of the training data. The batch size of 32 is also suitable for training the model in mini-batches, allowing for faster convergence and better memory efficiency.



**Models and their different Hyperparameters Configurations** 
To complete this course work 3 models were used, of which all gave different accuracies and loss." Model3" was chosen because it gave far better results than the others.  
 
Model 1
This model is a deep feedforward neural network with three dense layers of 512, 256 and 3 neurons, respectively. It uses a 'Flatten' layer to reshape the input data to a 1D array, and the output layer has a softmax activation function that outputs probabilities for each of the three classes
The model was compiled with 'adam' optimizer, 'categorical cross-entropy' loss function, and 'accuracy' as the performance metric.
When evaluated on Validation data, It gave an accuracy of 33% and a loss of 2.4  as can be seen in Figure below 
 
Model 2
This model is a convolutional neural network (CNN) that consists of two convolutional layers followed by a flatten layer and a dense output layer.

The input shape is (300,300,1), indicating the input is a grayscale image with a size of 300x300 pixels. The first convolutional layer has 64 filters with a kernel size of 3x3, followed by a ReLU activation function. The second convolutional layer has 32 filters with a kernel size of 3x3, also followed by a ReLU activation function.
When evaluated on Validation data, It gave an accuracy of 78% and a loss of 0.5  as can be seen in Figure below 

Model 3
The first convolutional layer has 64 filters with a kernel size of 3x3, followed by a ReLU activation function. The second convolutional layer has 32 filters with a kernel size of 3x3, also followed by a ReLU activation function. The third convolutional layer has 32 filters with a kernel size of 3x3 and again followed by a ReLU activation function.

The dropout layer is added to reduce overfitting by randomly dropping out 50% of the neurons in the previous layer during training.
When evaluated on Validation data, It gave an accuracy of 78% and a loss of 0.5  as can be seen in Figure below 

The Game
As instructed in the CW a game of a rock papers scissors of needs was require for extra credit.
To this a function  had to be written to take 3 arguments (2 images and the model) after which based on the set of if/else conditions a decision is made with the help of the model and a result is returned. 
Definition 


Conclusion
In conclusion, though my final model has a high accuracy rate in both train and test validation, more could still be done to improve the model, which is something I intend to do in my personal time. The CW opened me up to how some of the fundamentals taught during the semester could be applied to solve real-life problems in CV.
I also gained insight into how to changes in hyperparameters could make your model relatively better or worse.
And finally, the last part of developing the function for my rock paper scissor game gave me the opportunity to implement my model in a real-time. 
