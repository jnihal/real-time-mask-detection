import os
import tensorflow as tf
import numpy as np

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications import MobileNetV2

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.python.ops.gen_math_ops import mod


TEST_SIZE = 0.2
LEARNING_RATE = 0.0001
DATA_PATH = os.path.join(os.getcwd(), 'data')


def loadData():

    images = list()
    labels = list()

    for folder in os.listdir(DATA_PATH):
        if folder != '.DS_Store':
            folder_path = os.path.join(DATA_PATH, folder)
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                image = load_img(file_path, target_size=(224, 224))
                image = img_to_array(image)
                image = preprocess_input(image)

                images.append(image)
                labels.append(folder)

    le = LabelEncoder()
    le.fit(labels)
    labels = le.transform(labels)
    labels = tf.keras.utils.to_categorical(labels)

    images = np.array(images, dtype="float32")
    labels = np.array(labels)

    return train_test_split(images, labels, test_size=TEST_SIZE, stratify=labels)


def buildModel(save=False):

    X_train, X_test, y_train, y_test = loadData()

    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        weights='imagenet',
        include_top=False
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical')(inputs)
    x = tf.keras.layers.experimental.preprocessing.RandomRotation(0.2, fill_mode='nearest')
    x = tf.keras.layers.experimental.preprocessing.RandomContrast(0.2)
    x = tf.keras.layers.AveragePooling2D(pool_size=(7, 7))(base_model.output)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(2, activation='softmax')(x)

    model = tf.keras.models.Model(inputs=base_model.input, outputs=output)

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20)

    print(model.evaluate(X_test, y_test))

    if save:
        model.save('MaskDetect')


if __name__ == '__main__':
    buildModel(save=False)