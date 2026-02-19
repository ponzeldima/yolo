import os
import random
import numpy as np
from io import BytesIO
from utils import print_image, print_images, resize_image

# Plotting and dealing with images
import matplotlib.pyplot as plt

import tensorflow as tf

# Interactive widgets
from ipywidgets import widgets

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow_datasets as tfds


dataset, info = tfds.load(
    "cats_vs_dogs", 
    with_info=True, 
    as_supervised=True, 
    data_dir="./data/cats-dogs")
dataset: dict[str, tf.data.Dataset]
info: tfds.core.DatasetInfo

# print(info)
# print(info.splits)
# print(info.splits['train'].num_examples)


train_ds = dataset['train']
train_ds: tf.data.Dataset
# print("Image shape:", train_ds.take(1))


images_to_plot = []
labels = []
for image, label in train_ds.take(1000):
    images_to_plot.append(image)
    labels.append(label)
# print_images(images_to_plot)

resized_images = train_ds.map(resize_image, num_parallel_calls=tf.data.AUTOTUNE)
resized_images_to_plot = []
for image, label in resized_images.take(1000):
    resized_images_to_plot.append(image)

# print_images(resized_images_to_plot)

# print(resized_images_to_plot[0])
# print(resized_images_to_plot[0])

def split_data(images, labels):
    # shuffled = images.shuffle(buffer_size=10000, seed=42)
    # dataset_size = tf.data.experimental.cardinality(shuffled).numpy()
    images = np.array(images)
    labels = np.array(labels)
    dataset_size = len(list(images))  # Not efficient for large sets, see Option 2 below

    train_size = int(0.8 * dataset_size)
    val_size = int(0.9 * dataset_size)

    train_ds = images[:train_size]
    val_ds = images[train_size:val_size]
    test_ds = images[val_size:]
    train_labels = labels[:train_size]
    val_labels = labels[train_size:val_size]
    test_labels = labels[val_size:]
    
    return (train_ds, train_labels, val_ds, val_labels, test_ds, test_labels)
    
    train_images = []
    train_labels = []
    val_images = [] 
    val_labels = [] 
    test_images = []
    test_labels = []
    for image, label in tfds.as_numpy(train_ds):
        train_images.append(image)
        train_labels.append(label)
    
    for image, label in tfds.as_numpy(val_ds):
        val_images.append(image)
        val_labels.append(label)
    
    for image, label in tfds.as_numpy(test_ds):
        test_images.append(image)
        test_labels.append(label)
    
    return (
        np.array(train_images), 
        np.array(train_labels), 
        np.array(val_images), 
        np.array(val_labels), 
        np.array(test_images), 
        np.array(test_labels))

def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(150, 150, 3)),
        tf.keras.layers.Conv2D(16, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The second convolution
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        # The third convolution
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),

        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        # tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
        metrics=['accuracy']
    ) 
        
    return model

print(np.array(resized_images_to_plot).shape)
(train_images, train_labels, val_images, val_labels, test_images, test_labels) = split_data(resized_images_to_plot, labels)
print(train_images.shape)
print(train_labels.shape)
print(val_images.shape)
print(val_labels.shape)
print(test_images.shape)
print(test_labels.shape)

model = create_model()

model.fit(train_images, train_labels, epochs=20)


results = model.evaluate(val_images, val_labels)
print("Test loss, Test accuracy:", results)

