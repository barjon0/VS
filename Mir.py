# import librosa
# import sklearn
import os
import re
# from sklearn.ensemble import RandomForestClassifier
from torchvggish import vggish, vggish_input
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from pprint import pprint

# duration = ??
# sampling rate?
N, H = 4096, 1024
# TRAIN_MUSIC_DIR = "music/train"
# TEST_MUSIC_DIR="music/test"
MUSIC_DIR="music/music-wav"
EMBEDDING_MODEL = vggish()
EMBEDDING_MODEL.eval()
FEATURE_LENGTH = 128 # TODO: search way to do this automatically, instead of by trial-and-error


def readFolder2Embedding(path):

    directory = os.fsencode(path)
    songs = os.listdir(directory)

    # vggish output embedding has length 128
    data = np.zeros((len(songs), FEATURE_LENGTH*128))
    labels = []

    for i, song in enumerate(songs):
        print(f"Learning Embedding for {song}")
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
    # directories = sorted(os.listdir(TRAIN_MUSIC_DIR))
    directories = sorted(os.listdir(MUSIC_DIR))
    data = np.empty((0, FEATURE_LENGTH*128))
    label = []
    for directory in directories:
        dir_data, dir_label = readFolder2Embedding(
            MUSIC_DIR+"/"+directory)
            # TRAIN_MUSIC_DIR+"/"+directory)
        data = np.concatenate((data, dir_data), axis=0)
        label += dir_label
    print("All data read in successfully!")
    return data, label


def classify(classifier, filepath):
    print("Predicting label for" + filepath + "...")
    embedding = EMBEDDING_MODEL.forward(vggish_input.wavfile_to_examples(filepath))
    converted_embedding = embedding.detach().numpy()[
        :FEATURE_LENGTH, :].flatten()
    return classifier.predict(converted_embedding.reshape(1, -1))

def classifyBatch(classifier, directory_path):
    directory = os.fsencode(directory_path)
    songs = os.listdir(directory)
    res=[]
    for song in songs:
        embedding = EMBEDDING_MODEL.forward(vggish_input.wavfile_to_examples(directory_path+"/"+song.decode("utf-8")))
        converted_embedding = embedding.detach().numpy()[:FEATURE_LENGTH, :].flatten()
        predicted_class=classifier.predict(converted_embedding.reshape(1, -1))
        res.append({"song":song, "predicted_class": predicted_class[0]})
    return res

def test(classifier, data, labels):
    correct=0
    for i,sample in enumerate(data):
        predicted_class=classifier.predict(sample.reshape(1, -1))
        if predicted_class == labels[i]:
            correct+=1
    return correct/len(data)


if __name__ == "__main__":

    data, label = readIn()

    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.33, random_state=42)

    print("Training:")
    print(x_train)
    print(y_train)
    print("Testing:")
    print(x_test)
    print(y_test)
    classifier=DecisionTreeClassifier(max_depth=5)
    classifier.fit(x_train, y_train)
    print("Training successful")

    print("Accuracy:"+ test(classifier,x_test, y_test))


    # pprint(classifyBatch(classifier, TEST_MUSIC_DIR))

    
