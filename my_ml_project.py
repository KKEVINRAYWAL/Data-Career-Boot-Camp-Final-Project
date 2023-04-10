import os


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import os
import kaggle

import os
import kaggle
import os
import json
import kaggle
import zipfile
import logging

import os
import logging


from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator


import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def data_preprocessing(data_dir, img_size=224, test_size=0.2, random_state=42):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create a console handler and set its log level to INFO
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Create a formatter to specify the log message format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)

    # Add the console handler to the logger
    logger.addHandler(ch)

    if not os.path.isdir(data_dir):
        logger.error(f"{data_dir} is not a directory")
        return None
    
    # Traverse the directory and extract the file paths and labels
    data = []
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            for file_name in os.listdir(class_dir):
                if file_name.endswith('.jpg'):
                    file_path = os.path.join(class_dir, file_name)
                    data.append({'path': file_path, 'class': class_name})

    if len(data) == 0:
        logger.error(f"No image files found in {data_dir}")
        return None

    # Convert the Python list to a Pandas DataFrame
    df = pd.DataFrame(data)

    # Split dataset into training and testing sets
    try:
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    except Exception as e:
        logger.error(f"Error splitting dataset: {e}")
        return None

    # Define the ImageDataGenerator with the preprocessing techniques
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rescale=1./255,
        fill_mode='nearest'
    )

    # Generate the training and testing images and labels using the ImageDataGenerator
    train_generator = datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='path',
        y_col='class',
        target_size=(img_size, img_size),
        batch_size=32,
        class_mode='categorical'
    )
    test_generator = datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col='path',
        y_col='class',
        target_size=(img_size, img_size),
        batch_size=32,
        class_mode='categorical'
    )

    logger.info("Data preprocessing completed successfully")
    return train_generator, test_generator

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

import numpy as np
from sklearn.metrics import classification_report
import logging

def build_model(input_shape=(224, 224, 3), num_classes=19):
    # Set up logging
    logging.basicConfig(filename='model_building.log', level=logging.INFO)
    
    # Check input_shape and num_classes
    assert isinstance(input_shape, tuple), "input_shape must be a tuple"
    assert isinstance(num_classes, int), "num_classes must be an integer"
    assert len(input_shape) == 3, "input_shape must be a tuple of 3 integers"
    assert input_shape[0] > 0 and input_shape[1] > 0 and input_shape[2] > 0, "each element of input_shape must be greater than 0"
    assert num_classes > 0, "num_classes must be greater than 0"

    # Create a sequential model
    logging.info("Building the model architecture...")
    model = Sequential()
    
    # Add convolutional layers
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    
    # Flatten the output of the convolutional layers
    model.add(Flatten())
    
    # Add fully connected layers
    model.add(Dense(512, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    # Explicitly set the output shape of the final layer to num_classes
    model.layers[-1].output_shape = (None, num_classes)
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    logging.info("Model building complete!")
    return model


import logging
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def train_model(model, train_generator, test_generator, epochs=10):
    # Set up logging
    logging.basicConfig(filename='training.log', level=logging.INFO)

    # Check model
    assert isinstance(model, Sequential), "model must be a Sequential instance"

    # Create an image data generator with data augmentation for the training set
    train_datagen = ImageDataGenerator(rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       fill_mode='nearest')

    # Create an image data generator for the validation set
    val_datagen = ImageDataGenerator()

    # Train the model
    logging.info("Training the model...")
    history = model.fit(train_generator,
                        epochs=epochs,
                        validation_data=test_generator)

    # Save the model
    logging.info("Saving the trained model...")
    model.save('trained_model.h5')

    # Return the trained model
    logging.info("Training complete!")
    return model






import matplotlib.pyplot as plt

def plot_loss(history):
    # Get training and validation loss values
    training_loss = history.history['loss']
    validation_loss = history.history['val_loss']

    # Get number of epochs
    num_epochs = len(training_loss)

    # Set epoch numbers
    epoch_nums = range(1, num_epochs+1)

    # Plot training and validation loss over epochs
    plt.plot(epoch_nums, training_loss)
    plt.plot(epoch_nums, validation_loss)
    plt.title('Training and validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Log loss')
    plt.legend(['Training', 'Validation'], loc='upper right')
    plt.show()


from sklearn.metrics import classification_report

def evaluate_model(model, test_dir, batch_size=32):
    import os
    # Defensive programming: check if model is an instance of keras.models.Model
    if not isinstance(model, keras.models.Model):
        raise TypeError("model should be an instance of keras.models.Model")

    # Defensive programming: check if test_dir is a valid directory
    if not os.path.isdir(test_dir):
        raise ValueError("test_dir should be a valid directory")

    # Defensive programming: check if batch_size is a positive integer
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size should be a positive integer")

    # Create an image data generator for the test set
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Create a generator for the test set
    test_generator = test_datagen.flow_from_directory(test_dir,
                                                      target_size=(224, 224),
                                                      batch_size=batch_size,
                                                      class_mode='categorical',
                                                      shuffle=False)

    # Make predictions on the test set
    try:
        y_pred = model.predict_generator(test_generator, steps=test_generator.samples // batch_size + 1)
    except Exception as e:
        print("An error occurred while making predictions on the test set:", e)
        return
    
    y_pred = np.argmax(y_pred, axis=1)

    # Get the true labels for the test set
    y_true = test_generator.classes

    # Get the class names for the test set
    class_names = list(test_generator.class_indices.keys())

    # Print the classification report
    try:
        print(classification_report(y_true, y_pred, target_names=class_names))
    except Exception as e:
        print("An error occurred while printing the classification report:", e)

                                         
