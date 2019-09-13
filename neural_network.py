# Imports
import tensorflow as tf
from tensorflow.keras.applications import nasnet # Pretrained models
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, model_size="large"):
        self.model_size = model_size
        self.model = self.get_model()

    def get_model(self):
        if self.model_size == "large":
            return nasnet.NASNetLarge()
        elif self.model_size == "small":
            return nasnet.NASNetMobile()

    # To preprocess an image to transform it to what is required by the Neural Network
    def preprocess(self, image):
        if self.model_size == "large":
            height = 331
            width = 331
        elif self.model_size == "small":
            height = 224
            width = 224
        image_resized = tf.image.resize_with_pad(image, target_height=height, target_width=width)
        image_preprocessed = nasnet.preprocess_input(image_resized)
        image_with_batch = tf.expand_dims(image_preprocessed, axis=0)
        return image_with_batch

    # Given the path of an image, returns the neural network's predictions for that image
    def get_predictions(self, image_filename):
        image = plt.imread(image_filename)
        image_preprocessed = self.preprocess(image)
        probs = self.model.predict(image_preprocessed)
        preds = nasnet.decode_predictions(probs)
        return preds[0]