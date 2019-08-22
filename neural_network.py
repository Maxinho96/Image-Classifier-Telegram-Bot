# Imports
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.applications import nasnet # Pretrained models
import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import wordnet as wn # WordNet database to translate English labels to Italian
import nltk

class NeuralNetwork:
    def __init__(self):
        self.model = self.get_model()
        self.download_database()

    # To translate the labels
    def download_database(self):
        nltk.download("wordnet")
        nltk.download('omw')

    def translate(self, id):
        pos = id[0]
        offset = int(id[1:])
        synset = wn.synset_from_pos_and_offset(pos, offset)
        words = synset.lemma_names("ita")
        ita_label = words[0] if len(words) > 0 else synset.lemma_names()[0]
        return ita_label

    # To preprocess an image to transform it to what is required by the Neural Network
    def preprocess(self, image):
        image_resized = tf.image.resize_image_with_pad(image, target_height=331, target_width=331)
        image_preprocessed = nasnet.preprocess_input(image_resized)
        image_with_batch = tf.expand_dims(image_preprocessed, axis=0)

    def get_model(self):
        return nasnet.NASNetLarge()

    def get_predictions(self, image_filename):
        image = plt.imread(image_filename)
        image_preprocessed = self.preprocess(image)
        probs = self.model.predict(image_preprocessed)
        pred = nasnet.decode_predictions(probs)
        ita_labels = [(self.translate(id), prob) for id, _, prob in pred[0]]
        return ita_labels
