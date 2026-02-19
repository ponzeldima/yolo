import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def prin_image(img):
    plt.imshow(img)
    plt.colorbar()
    plt.show()

def reshape_and_normalize(images, items_number):
    images = images.reshape((items_number, 28, 28, 1))
    images = np.divide(images, 255)
    return images

def convolutional_model():
    model = tf.keras.models.Sequential([ 
		tf.keras.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ]) 
    model.compile(
		optimizer='adam',
		loss='sparse_categorical_crossentropy',
		metrics=['accuracy']
	)          
    return model


fmnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = fmnist.load_data()

# prin_image(training_images[0])
# prin_image(test_images[0])
print(test_labels[0])

training_images=reshape_and_normalize(training_images, 60000)
test_images=reshape_and_normalize(test_images, 10000)


model = convolutional_model()
model.fit(training_images, training_labels, epochs=5)
prediction = model.predict(np.array([test_images[0]]))
print(prediction)

model.evaluate(test_images, test_labels)