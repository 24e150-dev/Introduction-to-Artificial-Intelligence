import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels = []
    
    # Loop through each category folder (0 to NUM_CATEGORIES - 1)
    for category in range(NUM_CATEGORIES):
        category_path = os.path.join(data_dir, str(category))
        
        # Check if the directory exists to avoid errors
        if not os.path.isdir(category_path):
            continue
            
        # Loop through each image file in the category folder
        for filename in os.listdir(category_path):
            file_path = os.path.join(category_path, filename)
            
            # Read the image using OpenCV
            img = cv2.imread(file_path)
            
            # Ensure the file was read correctly (skips hidden files like .DS_Store)
            if img is not None:
                # Resize the image to the required dimensions
                img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                
                # Append the resized image and its label to the lists
                images.append(img_resized)
                labels.append(category)
                
    return (images, labels)


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    # Create a convolutional neural network
    model = tf.keras.models.Sequential([
        
        # Convolutional layer: Learn 32 filters using a 3x3 kernel
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        
        # Max-pooling layer: Reduce spatial dimensions using 2x2 pool size
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Second Convolutional layer: Learn 64 filters using a 3x3 kernel
        tf.keras.layers.Conv2D(
            64, (3, 3), activation="relu"
        ),
        
        # Second Max-pooling layer
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Flatten units: Transform 2D matrix into a 1D vector
        tf.keras.layers.Flatten(),
        
        # Hidden layer: Add a dense layer with 128 units and ReLU activation
        tf.keras.layers.Dense(128, activation="relu"),
        
        # Dropout layer: Randomly drop out 50% of nodes to prevent overfitting
        tf.keras.layers.Dropout(0.5),
        
        # Output layer: NUM_CATEGORIES units with softmax activation for probabilities
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])
    
    # Compile the neural network
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model


if __name__ == "__main__":
    main()
