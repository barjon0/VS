import librosa
# import sklearn
import os
from sklearn.ensemble import RandomForestClassifier

#duration = ??
# sampling rate?
N, H = 4096, 1024

def readFolder2Waves(path):
    xList = list()
    directory = os.fsencode(path)
    for file in os.listdir(directory):
        xList.append(librosa.load(file))        #duration

    return xList

def readIn():
    xListJo = readFolder2Waves("VS/music/jonas")
    xListPh = readFolder2Waves("VS/music/philipp")

    featureListJo = getFeatures(xListJo)
    featureListPh = getFeatures(xListPh)
    
    data1 = [(x, 0) for x in featureListJo]
    data2 = [(x, 1) for x in featureListPh]


    label = [0]*len(xListJo)+[1]*len(xListPh)
    data = xListJo + xListPh

    return data, label

def getFeatures(waves):
    featureList = list()
    for w in waves:
        featureList.append(librosa.feature.chroma_stft(w))
    return None

def trainClassifier(classifier,trainingData):
    data, label= trainingData
    for i in range(len(data)):
        classifier.fit()
    pass
    
def classify(classifier,test):
    return classifier.predict(test)


if __name__ == "__main__":

    data, label = readIn()
    print(data)
    print(label)
    # classifier = RandomForestClassifier(n_estimators=10)
    #readIn
    #getFeatures
    # train
    # classify