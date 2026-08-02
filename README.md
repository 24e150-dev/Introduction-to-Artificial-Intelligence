Traffic Sign Classification - Experimentation Log
Overview
This project uses OpenCV and TensorFlow/Keras to build a Convolutional Neural Network (CNN) capable of classifying photographs of traffic signs into 43 different categories.

Experimentation Process
Attempt 1: The Baseline Model
I started with a relatively simple architecture to establish a baseline. The model consisted of a single Convolutional layer (32 filters, 3x3 kernel) followed by a Max-Pooling layer (2x2), a Flatten layer, a Dense hidden layer with 128 units, and an output layer.

Result: The model trained very quickly and achieved high accuracy on the training set, but performed poorly on the testing set. This was a clear sign of overfitting—the model was memorizing the training images rather than learning general features of traffic signs.

Attempt 2: Combating Overfitting with Dropout
To address the overfitting issue, I introduced a Dropout layer with a rate of 0.5 (randomly dropping half of the neural connections) immediately after the hidden Dense layer.

Result: The gap between training accuracy and testing accuracy narrowed significantly. The model was generalizing much better, but the overall accuracy stalled at around 85%. I realized the network wasn't deep enough to capture the complex shapes and edges of 43 different signs.

Attempt 3: Adding Depth (The Final Architecture)
To help the model learn more complex visual hierarchies, I added a second Convolutional layer (increasing the filters to 64) and a second Max-Pooling layer right after the first set. This allows the first layer to learn simple edges and colors, while the second layer learns distinct shapes (like octagons or arrows).

Result: This was the breakthrough. Accuracy on the testing set jumped to over 95%. The combination of two Convolutional/Pooling blocks extracted enough detailed features, while the 0.5 Dropout layer successfully prevented overfitting.

Final Model Architecture
After experimenting with different pool sizes and filter amounts, the final optimized model uses the following structure:

Conv2D: 32 filters, 3x3 kernel, ReLU activation

MaxPooling2D: 2x2 pool size

Conv2D: 64 filters, 3x3 kernel, ReLU activation

MaxPooling2D: 2x2 pool size

Flatten: Converts 2D feature maps to 1D vectors

Dense: 128 units, ReLU activation

Dropout: 0.5 rate to prevent overfitting

Dense (Output): 43 units (NUM_CATEGORIES), Softmax activation