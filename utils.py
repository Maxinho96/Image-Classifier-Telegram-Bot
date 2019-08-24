from nltk.corpus import wordnet as wn # WordNet database to translate English labels to Italian
import nltk

class Utils:
    def __init__(self):
        self.download_database()

    # Downloads the WordNet database
    def download_database(self):
        nltk.download("wordnet")
        nltk.download('omw')

    # Converts a list of neural network predictions to an output string. Each prediction should be a tuple of three
    # elements: a WordNet offset id, a label name and the neural network's confidence (probability) of that prediction
    def preds_to_string(self, preds):
        output = ""
        for id, label, prob in preds:
            output += "*-* _Eng:_*{}*, _Ita:_*{}* ({:.2f}%)\n".format(label, self.id_to_ita(id), prob * 100)
        return output

    # Gets a WordNet offset id and returns the corresponding italian word
    def id_to_ita(self, id):
        pos = id[0]
        offset = int(id[1:])
        synset = wn.synset_from_pos_and_offset(pos, offset)
        words = synset.lemma_names("ita")
        ita_label = words[0] if len(words) > 0 else "[Non disponibile]"
        return ita_label