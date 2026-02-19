import matplotlib.pyplot as plt
import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def print_image(img):
    plt.imshow(img)
    plt.colorbar()
    plt.show()
    
def print_images(images, titles=None, cols=3):
    """
    Display multiple images in one window.
    :param images: list or batch of image tensors or arrays
    :param titles: optional list of titles (same length as images)
    :param cols: number of columns in the grid
    """
    n = len(images)
    rows = (n + cols - 1) // cols  # calculate rows automatically
    plt.figure(figsize=(4 * cols, 4 * rows))

    for i, img in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        if titles:
            plt.title(titles[i])
        plt.axis("off")

    plt.tight_layout()
    plt.show()
    
def resize_image(image, label, size=(150,150)):
    image = tf.image.resize(image, size)  # resize to 200x200
    image = tf.cast(image, tf.float32) / 255.0  # normalize if you want
    return image, label