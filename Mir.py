import librosa
# import sklearn
import os
import re
# from sklearn.ensemble import RandomForestClassifier
from torchvggish import vggish, vggish_input
# from vggish.vggish_input import wavfile_to_examples
# duration = ??
# sampling rate?
N, H = 4096, 1024
MUSIC_DIR = "music"


def readFolder2Waves(path):
    data= []
    labels = []
    directory = os.fsencode(path)
    embedding_model = vggish()
    embedding_model.eval()
    for file in os.listdir(directory):
        # data.append(librosa.load(path+"/"+file.decode("utf-8")))  # normalize duration?
        # data.append(wavfile_to_examples(path+"/"+file.decode("utf-8")))  # normalize duration?
        embeddings = embedding_model.forward(vggish_input.wavfile_to_examples(path+"/"+file.decode("utf-8")))
        data.append(embeddings) 
        labels.append(re.match(r".*cat_(.*)",path)[1])

    return data, labels


def readIn():
    directories = sorted(os.listdir(MUSIC_DIR))
    data= []
    label= []
    for directory in directories:
        dir_data, dir_label  =  readFolder2Waves(MUSIC_DIR+"/"+directory)
        data.append(dir_data)
        label.append(dir_label)
    return data, label


def getFeatures(waves):
    featureList = list()
    for w in waves:
        featureList.append(librosa.feature.chroma_stft(w))
    return None


def trainClassifier(classifier, trainingData):
    data, label = trainingData
    for i in range(len(data)):
        classifier.fit()
    pass


def classify(classifier, test):
    return classifier.predict(test)


if __name__ == "__main__":

    data, label = readIn()
    print(data)
    print(label)
    # classifier = RandomForestClassifier(n_estimators=10)
    # readIn
    # getFeatures
    # train
    # classify
