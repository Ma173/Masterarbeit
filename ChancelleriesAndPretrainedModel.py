import gensim
from gensim import utils
from gensim.test.utils import datapath
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np
# import nltk
# import xgboost as xgb
# import unicodedata


def preprocessing(corpus_path):
    chancelleryBlocksRaw = open(datapath(corpus_path), encoding="utf-8").read().split("__________")
    chancelleryBlocks = []

    # Erstellung einer Liste mit allen Kanzlei-Blöcken
    # und je BLock einer Liste aller Zeilen, je Zeile eine Liste aller Einheiten (Feature-Nummer, Fundort, etc.)

    # Pro Kanzlei-Block
    global currentChancelleryName, currentChancelleryHTML, currentLine, currentLineSplitted, newChancelleryBlock
    for i in range(len(chancelleryBlocksRaw)):
        currentChancelleryBlockSplitted = []
        currentChancelleryBlock = chancelleryBlocksRaw[i].splitlines()
        newChancelleryBlock = []

        # Pro Zeile in Kanzlei-Block
        for k in range(len(currentChancelleryBlock)):
            currentLine = currentChancelleryBlock[k]
            currentLineSplitted = []

            if currentLine.startswith("F"):
                featureInfo = currentLine[:currentLine.find(" |")].split(" ")
                featureInfoActualData = currentLine[currentLine.find(" |"):]
                tokens = utils.simple_preprocess(featureInfoActualData)
                words = [word.lower() for word in tokens if word.isalpha()]
                from nltk.corpus import stopwords
                stopWords = set(stopwords.words('german'))

                words = [word for word in words if not word in stopWords]

                for featureInfoElement in featureInfo:
                    currentLineSplitted.append(featureInfoElement)

                currentLineSplitted.append(words)  # (featureInfoActualData)
                newChancelleryBlock.append(currentLineSplitted)
            elif currentLine.startswith("<"):
                currentChancelleryHTML = currentLine
            else:
                currentChancelleryName = currentLine
                if currentChancelleryName: newChancelleryBlock.append(currentChancelleryName)

        chancelleryBlocks.append(newChancelleryBlock)  # ,currentChancelleryHTML])
    return chancelleryBlocks


# TODO: Dran denken, dass die Elemente 5 & 6 in den Kanzleizeilen optional sind, weil 3 & 4 = 5 & 6 sein können (Textquelle = Fundstelle)
mainData = preprocessing(r'C:/Users/Malte/Dropbox/Studium/Linguistik Master HHU/Masterarbeit/websitesTextsReformatted.txt')
for block in mainData:
    print(block)

# Load word2vec model (trained on an enormous Google corpus)
# TODO Hier das Wikipedia-Model, nachdem ich es runtergeladen habe, einbinden
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# Check dimension of word vectors
model.vector_size
