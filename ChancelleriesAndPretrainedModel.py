import pandas as pd
import gensim
# import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# import xgboost as xgb
from gensim.test.utils import datapath

chancelleryBlocks = []


def preprocessing(corpus_path):
    chancelleryBlocksRaw = open(datapath(corpus_path), encoding="utf-8").read().split("__________")
    # print(chancelleryBlocksRaw[0])

    # Erstellung einer Liste mit allen Kanzlei-Blöcken
    # und je BLock einer Liste aller Zeilen, je Zeile eine Liste aller Einheiten (Feature-Nummer, Fundort, etc.)

    # Pro Kanzlei-Block
    global currentChancelleryName, currentChancelleryHTML, currentLine, currentLineSplitted, newChancelleryBlock
    for i in range(len(chancelleryBlocksRaw)):
        # print("Preprocessing chancelleryBlock", i, "of", len(chancelleryBlocksRaw))
        currentChancelleryBlockSplitted = []
        currentChancelleryBlock = chancelleryBlocksRaw[i].splitlines()
        newChancelleryBlock = []
        # Pro Zeile in Kanzlei-Block
        for k in range(len(currentChancelleryBlock)):
            # print("\tPreprocessing line", k, "of", len(currentChancelleryBlock))
            currentLine = currentChancelleryBlock[k]
            currentLineSplitted = []

            # currentLineUntilExactText = currentLine[:currentLine.find(" ", currentLine.find(" ") + 6)]
            if currentLine.startswith("F"):
                # print([i for i, c in enumerate(currentLine) if c == " "][6 - 1])
                # currentLineSplitted = currentLine.split(" ")[:[i for i, c in enumerate(currentLine) if c == " "][6 - 1]]
                # currentLineSplitted.append(currentLine[[i for i, c in enumerate(currentLine) if c == " "][6 - 1] + 1:])
                featureInfo = currentLine[:currentLine.find(" |")].split(" ")
                featureInfoPartTwo = "PLATZHALTER" # currentLine[currentLine.find(" |"):]
                for featureInfoElement in featureInfo:
                    currentLineSplitted.append(featureInfoElement)
                # newChancelleryBlock.append(currentLineSplitted)
                currentLineSplitted.append(featureInfoPartTwo)

                # print("Feature gefunden:",featureInfo)
                # print("Feature Part 2:",featureInfoPartTwo)
                # print(currentLineSplitted)
                newChancelleryBlock.append(currentLineSplitted)
            elif currentLine.startswith("<"):
                currentChancelleryHTML = currentLine
                # newChancelleryBlock.append(currentLineSplitted)
            else:
                currentChancelleryName = currentLine
                if (currentChancelleryName):
                    newChancelleryBlock.append(currentChancelleryName)

        chancelleryBlocks.append(newChancelleryBlock)  # ,currentChancelleryHTML])


# TODO: Dran denken, dass die Elemente 5 & 6 in den Kanzleizeilen optional sind, weil 3 & 4 = 5 & 6 sein können (Textquelle = Fundstelle)
preprocessing(r'C:/Users/Malte/Dropbox/Studium/Linguistik Master HHU/Masterarbeit/websitesTextsReformatted.txt')
# currentLineUntilExactText = currentLine[:currentLine.find(" ", currentLine.find(" ") + 10)]
# currentLineUntilExactText = currentLine.find(' ', firstTest + 4)
# currentLineUntilExactText= [i for i, c in enumerate(currentLine) if c == " "][6-1]
# print(chancelleryBlocks)
for block in chancelleryBlocks:
    print(block)

