import os
import tensorflow as tf
import numpy as np

from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications import MobileNetV2

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


TEST_SIZE = 0.2
LEARNING_RATE = 0.0001
DATA_PATH = os.path.join(os.getcwd(), 'data')


def loadData():

    # List to store images and labels
    images = list()
    labels = list()

    # Read the dataset
    for folder in os.listdir(DATA_PATH):
        if folder != '.DS_Store':
            folder_path = os.path.join(DATA_PATH, folder)
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)

                # Read and process the image
                image = load_img(file_path, target_size=(224, 224))
                image = img_to_array(image)
                image = preprocess_input(image)

                # Add the image and corresponding label
                images.append(image)
                labels.append(folder)

    # Convert labels to numeric form
    le = LabelEncoder()
    le.fit(labels)
    labels = le.transform(labels)
    labels = tf.keras.utils.to_categorical(labels)

    # Convert the list as a numpy array
    images = np.array(images, dtype="float32")
    labels = np.array(labels)

    # Return the training and testing set
    return train_test_split(images, labels, test_size=TEST_SIZE, stratify=labels)


def buildModel(save=False):

    # Get the training and testing data
    X_train, X_test, y_train, y_test = loadData()

    # Use the MobileNetV2 model for transfer learning
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        weights='imagenet',
        include_top=False
    )
    base_model.trainable = False
    

    # Data augmentation
    data_augmentation = ImageDataGenerator(
        rotation_range=50,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    # Add custom layers to the base model
    x = tf.keras.layers.AveragePooling2D(pool_size=(7, 7))(base_model.output)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)

    # Build the model
    model = tf.keras.models.Model(inputs=base_model.input, outputs=outputs)

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Fit data to the model with data augmentation
    model.fit(data_augmentation.flow(X_train, y_train), epochs=20)

    # Evaluate model on the testing set
    print(model.evaluate(X_test, y_test))

    # Save the model
    if save:
        model.save('MaskDetect')


if __name__ == '__main__':
    buildModel(save=False)