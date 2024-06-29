# import librosa
# import sklearn
import os
import re
# from sklearn.ensemble import RandomForestClassifier
from torchvggish import vggish, vggish_input
from sklearn import svm
import numpy as np

# duration = ??
# sampling rate?
N, H = 4096, 1024
TRAIN_MUSIC_DIR = "music/train"

EMBEDDING_MODEL = vggish()
EMBEDDING_MODEL.eval()
FEATURE_LENGTH = 218


def readFolder2Embedding(path):

    directory = os.fsencode(path)
    songs = os.listdir(directory)

    # vggish output embedding has length 128
    data = np.zeros((len(songs), FEATURE_LENGTH*128))
    labels = []

    for i, song in enumerate(songs):
        embedding = EMBEDDING_MODEL.forward(
            vggish_input.wavfile_to_examples(path+"/"+song.decode("utf-8")))
        # normalize length, convert to numpy array and flatten the feature array
        converted_embedding = embedding.detach().numpy()[
            :FEATURE_LENGTH, :].flatten()
        data[i, :] = converted_embedding
        # get label/category name from directory name
        labels.append(re.match(r".*cat_(.*)", path)[1])

    return data, labels


def readIn():
    directories = sorted(os.listdir(TRAIN_MUSIC_DIR))
    data = np.empty((0, FEATURE_LENGTH*128))
    label = []
    for directory in directories:
        dir_data, dir_label = readFolder2Embedding(
            TRAIN_MUSIC_DIR+"/"+directory)
        data = np.concatenate((data, dir_data), axis=0)
        label += dir_label
    return data, label


def classify(classifier, path):
    embedding = EMBEDDING_MODEL.forward(vggish_input.wavfile_to_examples(path))
    converted_embedding = embedding.detach().numpy()[
        :FEATURE_LENGTH, :].flatten()
    return classifier.predict(converted_embedding.reshape(1, -1))


if __name__ == "__main__":

    data, label = readIn()

    classifier = svm.SVC()
    classifier.fit(data, label)

    # print(classify(classifier, ""))
