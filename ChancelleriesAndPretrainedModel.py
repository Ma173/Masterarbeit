import pickle
import random
import time
from datetime import datetime
import gensim
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
import winsound
from bs4 import BeautifulSoup
from gensim import utils
from gensim.models import KeyedVectors, Doc2Vec
from gensim.test.utils import datapath
from joblib import Parallel
from matplotlib import pyplot as plt
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from sklearn.feature_extraction.text import TfidfVectorizer
import json

from sklearn.metrics import recall_score, precision_score
from sklearn.model_selection import cross_val_score

stopWords = set(stopwords.words('german'))
# import nltk
# import xgboost as xgb
# import unicodedata

mainDataWordsList = []
chancelleriesFeatureExpressionsWordsList = []
global chancelleryHTMLtexts
chancelleryHTMLtexts = []
chancelleriesWordDensites = []
chancelleriesSentences = {}
wordCountsCumulated = {}
lemmaCountsPerChancellery = {}
nlp = spacy.load('de_core_news_sm')
lemmatizer = nltk.stem.WordNetLemmatizer()
germanStopWords = stopwords.words('german')
chancelleriesPosTagCounts = {}
chancelleriesSyntacticDependencies = {}


def preprocessing(corpus_path):
    valuableFeatureIds = ["F8", "F22", "F23", "F24"]

    with open(datapath(corpus_path), 'r', encoding='utf-8') as fileToLoad:  # former: encoding='unicode_escape'
        mainDataRaw = fileToLoad.read()

    mainDataRaw = mainDataRaw.rstrip('\n')  # mainDataRaw = mainDataRaw.replace("\n", "")
    # TODO: Wie entferne ich alle Newline-Characters ("\n"), aber NICHT die regulären Zeilenumbrüche?

    chancelleryBlocksRaw = mainDataRaw.split("__________")  # vorher: encoding="utf-8"
    print("After splitting at the underscore marking the main data contains", len(chancelleryBlocksRaw), "chancellery blocks")

    def print_chancellery_blocks():
        for block in chancelleryBlocksRaw:
            print("\n\n")
            for line in block:
                print(line)

    def print_specific_chancellery_block(chancelleryName):
        for i, block in enumerate(chancelleryBlocksRaw):
            print(i)
            if block[0].strip() == chancelleryName:
                print("\n\n")
                for blockline in block:
                    print(blockline)

    # chancelleryBlocksRaw = open(datapath(corpus_path), encoding='unicode_escape').read().split("__________")  # vorher: encoding="utf-8"

    chancelleryBlocks = []

    # Erstellung einer Liste mit allen Kanzlei-Blöcken
    # und je BLock einer Liste aller Zeilen, je Zeile eine Liste aller Einheiten (Feature-Nummer, Fundort, etc.)

    global currentChancelleryHTML, currentLineSplitted, newChancelleryBlock, currentLineInfo

    # For every chancellery block
    for i, chancelleryBlock in enumerate(chancelleryBlocksRaw):
        currentChancelleryBlock = chancelleryBlock.splitlines()
        newChancelleryBlock = []
        currentChancelleryWordsList = []
        currentHTMLwordsList = []
        currentFeatureExpressions = []
        currentChancelleryName = ""
        averageSentenceLength = 0
        averageWordDensity = 0
        numberOfSentences = 0
        currentChancelleryLemmaCount = {}
        currentChancelleryPosTagCount = {}
        chancellerySyntacticDependencies = []
        # if i == 23:
        #     print("ChancelleryBlock", i, "consists of", len(currentChancelleryBlock), "lines.")
        #     print("currentChancelleryBlock 23:\n", currentChancelleryBlock)

        startTimerProcessingChancelleryBlock = time.time()
        timeForProcessingFeatureLines = 0
        timeForProcessingHtmlLine = 0
        timeForParsingHtml = 0
        timeForCleaningWordsOfDebris = 0
        timeForNlpAndPosTagging = 0
        timeForCountingLemmas = 0

        # For every line in the current chancellery block
        for k in range(len(currentChancelleryBlock)):
            currentLine = currentChancelleryBlock[k]

            # If the current line is a list of lines (strings) that start with "F"
            if currentLine.startswith("F", 0, 1):
                currentLineInfo = "Feature_line"
                startTimerProcessingFeatureLines = time.time()
                currentLineSplitted = []
                currentFeatureId = currentLine.split(" ")[0]
                featureInfo = currentLine[:currentLine.find(" |")].split(" ")
                featureInfoActualData = currentLine[currentLine.find(" |"):]

                # Checking for the current feature id if it's a valuable one, then preprocessing the actual feature data
                if currentFeatureId in valuableFeatureIds:  # currentLine.startswith("F"):
                    tokens = utils.simple_preprocess(featureInfoActualData)
                    words = [word.lower() for word in tokens if word.isalpha()]
                    words = [word for word in words if not word in stopWords]
                    for word in words:
                        # mainDataWordsList.append(word)
                        currentChancelleryWordsList.append(word)
                    for featureInfoElement in featureInfo:
                        currentLineSplitted.append(featureInfoElement)
                    currentLineSplitted.append(words)  # (featureInfoActualData)
                    newChancelleryBlock.append(currentLineSplitted)
                    currentFeatureExpressions.append([currentLine.split(" ")[0], currentLine.split(" ")[1]])
                timeForProcessingFeatureLines += round((time.time() - startTimerProcessingFeatureLines), 2)

            # If the current line is a line that consists of HTML text (starting with "<")
            elif currentLine.startswith("<", 0, 1):
                currentLineInfo = "HTML_line"
                startTimerProcessingHtmlLine = time.time()
                currentChancelleryHTML = currentLine
                htmlCodesToReplaceWithSpace = ["<br>", "</br>", "<p>", "</p>"]
                currentChancelleryHTML = currentChancelleryHTML.replace("<br>", " ")
                currentChancelleryHTMLclean = BeautifulSoup(currentChancelleryHTML, "html.parser").get_text()
                timeForParsingHtml += round((time.time() - startTimerProcessingHtmlLine), 2)

                currentWordDensities = []  # The list of all sentences' word density in the current document
                currentChancelleryHTMLclean = currentChancelleryHTMLclean.replace("\t", " ")
                currentChancelleryHTMLclean = currentChancelleryHTMLclean.replace("  ", "")
                currentSentences = currentChancelleryHTMLclean.split(".")  # TODO: Prüfen, dass hier die Entfernung des Leerzeichens nicht mehr Probleme gemacht hat.
                currentSentencesAfterThreshold = []
                minimumSentenceLength = 5

                # Keeping only those sentences that pass the threshold of minimum sentence length
                for sentence in currentSentences:
                    if len(sentence) >= minimumSentenceLength:
                        currentSentencesAfterThreshold.append(" " + sentence)

                chancellerySentencesCleaned = []
                for sentence in currentSentencesAfterThreshold:
                    # sentenceWithoutSpecialChars = utils.simple_preprocess(sentence)

                    # sentenceTokenizedWithoutStopwords = [word for word in word_tokenize(sentence) if word.lower() not in germanStopWords]
                    # sentenceTokenized = remove_stopwords(sentenceWithoutSpecialChars)
                    # words = sentenceTokenized.split(" ")
                    wordsCleaned = []
                    for word in sentence.split(" "):
                        startTimerCleaningWordsOfDebris = time.time()

                        # If the current word doesn't consist of more than x Uppercases & is longer than 1, continue with the word
                        # and remove any leftovers of the HTML codes or any special characters
                        if not sum(1 for c in word if c.isupper()) > 3 and len(word) >= 2:
                            htmlDebris = ["\t", "\t\t", "\\xa0", "\xa0", "[", "]", "'", ">>", "<<", "|", "\\u00fcber", '"', "...", "\u200b", "\\u200b"]
                            wordToAppend = word
                            wordSplitted = ""
                            for debris in htmlDebris:
                                if debris in wordToAppend:
                                    wordToAppend = wordToAppend.replace(debris, "")
                            if "\\u00e4" in word:
                                wordToAppend = wordToAppend.replace("\\u00e4", "ä")
                            elif "\u00e4" in word:
                                wordToAppend = wordToAppend.replace("\u00e4", "ä")
                            # If the current word contains "." or "/" and these special chars are not at first or last position in the word
                            # then split the current word at this character. If that makes a list of two strings and both are not empty
                            # then append it to the current words list
                            specialChars = [".", "/", "?", ":", "!", "?"]
                            for specialChar in specialChars:
                                if specialChar in wordToAppend and wordToAppend.find(specialChar) != len(wordToAppend) and wordToAppend.find(specialChar) != 0:
                                    wordSplitted = wordToAppend.split(specialChar)

                            if len(wordSplitted) == 2:
                                if wordSplitted[0] != "":
                                    wordsCleaned.append(wordSplitted[0])
                                if wordSplitted[1] != "":
                                    wordsCleaned.append(wordSplitted[1])
                            if wordToAppend and wordToAppend[0].isdigit():
                                wordConsistingOfNumbersAndLetters = False
                                for char in wordToAppend[1:]:
                                    if not char.isalpha():
                                        wordConsistingOfNumbersAndLetters = True
                                if wordConsistingOfNumbersAndLetters:
                                    for m, char in enumerate(wordToAppend):
                                        if char.isalpha():
                                            number_part = word[:m]
                                            letter_part = word[m:]
                                            wordsCleaned.append(number_part)
                                            wordsCleaned.append(letter_part)

                            else:
                                wordsCleaned.append(wordToAppend)

                            #####
                            # pos-tagging & lemmatizing the sentence so its words can more easily be counted and compared to word lists
                            #####
                            # print("Starting pos tagging & lemmatizing process of chancellery", currentChancelleryName)

                            # re-align the cleaned words as a sentence, so it can be pos-tagged
                            sentenceCleaned = ""
                            for m, wordCleaned in enumerate(wordsCleaned):
                                if len(sentenceCleaned) > 0:
                                    sentenceCleaned += " " + wordCleaned
                                else:
                                    sentenceCleaned += wordCleaned

                            timeForCleaningWordsOfDebris += round((time.time() - startTimerCleaningWordsOfDebris), 2)

                            startTimerNlpAndPosTagging = time.time()
                            # Processing the sentence with the German language model of spacy
                            doc = nlp(sentenceCleaned)
                            lemmasWithPartsOfSpeech = []

                            # Iterating over all tokens in the cleaned sentence and saving the part of speech
                            for token in doc:
                                word = token.text
                                lemma = token.lemma_
                                partOfSpeechTag = token.pos_
                                currentChancelleryPosTagCount[partOfSpeechTag] = currentChancelleryPosTagCount.get(partOfSpeechTag, 0) + 1
                                lemmasWithPartsOfSpeech.append([lemma, partOfSpeechTag])
                                tokenWithSyntacticDependencies = {
                                    "text": token.text,
                                    "pos": token.pos_,
                                    "dep": token.dep_,
                                    "head": token.head.text
                                }
                                # if i == 0 and k < 2:
                                #    print(token.text, token.dep_, token.head.text, token.head.pos_, [child for child in token.children])
                                chancellerySyntacticDependencies.append(tokenWithSyntacticDependencies)
                            timeForNlpAndPosTagging += round((time.time() - startTimerNlpAndPosTagging), 2)

                            startTimerCountLemmas = time.time()
                            # Appending all cleaned words to the HTMLwordsList, the mainDataWordsList and the dictionary that counts the word count
                            # print("Appending all cleaned words to the HTMLwordsList, the mainDataWordsList and the dictionary that counts the word count")
                            for lemmaWithPos in lemmasWithPartsOfSpeech:
                                lemma = lemmaWithPos[0]
                                partOfSpeechTag = lemmaWithPos[1]
                                lemmaGroup = (lemma, partOfSpeechTag)
                                test = True
                                if lemma.lower().isalpha() and lemma.lower() not in germanStopWords:
                                    currentHTMLwordsList.append(lemma)  # [word.lower() for word in HTMLtokens if word.isalpha()]
                                    mainDataWordsList.append(lemma)
                                    if lemma in wordCountsCumulated:
                                        wordCountsCumulated[lemma] += 1
                                    else:
                                        wordCountsCumulated[lemma] = 1

                                    # Adding the cleaned word accompanied by its part of speech to the chancellery's word count dictionary
                                    # The style is: word : [[wordCountOfFirstOccurence, partOfSpeechOfFirstOccurence], [wordCountOfSecondOccurence, partOfSpeechOfSecondOcc...]
                                    # This way any ambiguity is saved if the part of speech of the current cleaned word is not the same as saved in the dict for this word

                                    # So if the lemmaGroup is already present in the dictionary
                                    if lemmaGroup in currentChancelleryLemmaCount:
                                        currentChancelleryLemmaCount[lemmaGroup] += 1
                                    else:
                                        currentChancelleryLemmaCount[lemmaGroup] = 1

                            timeForCountingLemmas += round((time.time() - startTimerCountLemmas), 4)
                            # print("Finished. Time elapsed:", timeSinceNlp, "seconds.")

                    # Carrying on with working on the former used wordsCleaned which seems to be fine until now (no lemmas needed yet):
                    # Removing any empty strings from the list of remaining words. These empty strings could have come up from previous cleaning processes
                    while "" in wordsCleaned:
                        wordsCleaned.remove("")

                    # Checking that the list is not empty and if the last sentence of the list is shorter than 4 words
                    # If that's the case, these words are appended to the previous list element (= the last sentence).
                    if chancellerySentencesCleaned and len(chancellerySentencesCleaned[-1]) < 4:
                        chancellerySentencesCleaned[-1] += wordsCleaned
                    else:
                        # If sentence length is greater or equal to the threshold of minimumSentenceLength
                        if len(wordsCleaned) >= minimumSentenceLength:
                            # If sentence isn't already in the list
                            if wordsCleaned not in chancellerySentencesCleaned:
                                # filter out all sentences that are above the maxConsecutiveUpperWords count
                                maxConsecutiveUpperWords = 3
                                maxcount = 0
                                consecutiveCount = 0
                                for m, word in enumerate(wordsCleaned):
                                    if word[0].isupper():
                                        consecutiveCount += 1
                                    if m == len(wordsCleaned) - 1 or not word[0].isupper():
                                        if maxcount < consecutiveCount:
                                            maxcount = consecutiveCount
                                        consecutiveCount = 0
                                if maxcount <= maxConsecutiveUpperWords:  # if not len([word for word in words if word[0].isupper()]) > maxcount:
                                    chancellerySentencesCleaned.append(wordsCleaned)
                                    currentWordDensities.append(len(wordsCleaned))
                # Filtering out any sentences that came in doubled
                chancellerySentencesToAppend = []
                for j, sentence in enumerate(chancellerySentencesCleaned):
                    if j == 0 or (j > 0 and sentence != chancellerySentencesCleaned[j - 1]):
                        chancellerySentencesToAppend.append(sentence)
                chancelleriesSentences[currentChancelleryName] = chancellerySentencesToAppend
                numberOfSentences = len(chancellerySentencesToAppend)
                try:
                    averageWordDensity = sum(currentWordDensities) / numberOfSentences
                except ZeroDivisionError:
                    print("\nDivision by zero. Number of sentences:", numberOfSentences, "| Chancellery is", currentChancelleryName, "| Sentences are",
                          chancellerySentencesToAppend)
                timeForProcessingHtmlLine += round((time.time() - startTimerProcessingHtmlLine), 2)

            elif currentLine.isalpha():
                currentLineInfo = "ChancelleryName_line"
                newChancelleryBlock.append(currentLine)
                currentChancelleryName = currentLine
                print("\nPreprocessing chancellery block", i, "of", len(chancelleryBlocksRaw), "|", currentChancelleryName)

        if averageWordDensity != 0:
            chancelleriesWordDensites.append(averageWordDensity)
        chancelleryCalculations = {"averageWordDensity": averageWordDensity}

        chancelleriesSyntacticDependencies[currentChancelleryName] = chancellerySyntacticDependencies
        chancelleriesPosTagCounts[currentChancelleryName] = currentChancelleryPosTagCount
        lemmaCountsPerChancellery[currentChancelleryName] = currentChancelleryLemmaCount
        # currentAverageWordDensityCompared = sum(chancelleriesWordDensites) / len(chancelleriesWordDensites)
        chancelleryBlocks.append([newChancelleryBlock, currentHTMLwordsList])  # currentChancelleryHTMLclean])  # ,currentChancelleryHTML])
        chancelleriesFeatureExpressionsWordsList.append([currentChancelleryName, currentChancelleryWordsList])
        chancelleryHTMLtexts.append([currentChancelleryName, currentHTMLwordsList, currentFeatureExpressions, chancelleryCalculations])  # currentChancelleryHTMLclean])

        timeSinceProcessingChancelleryBlock = round((time.time() - startTimerProcessingChancelleryBlock), 2)
        print("Finished processing chancellery", currentChancelleryName, "| Time elapsed:", timeSinceProcessingChancelleryBlock, "seconds.")
        # print("\tProcessing feature lines:", timeForProcessingFeatureLines, "seconds.")
        print("\tProcessing HTML line:", timeForProcessingHtmlLine, "seconds.")
        print("\t\tParsing the HTML line:", timeForParsingHtml, "seconds.")
        print("\t\tCleaning the words of debris:", timeForCleaningWordsOfDebris, "seconds.")
        print("\t\tNLP and pos-tagging:", timeForNlpAndPosTagging, "seconds.")
        print("\t\tCounting the lemmas:", timeForCountingLemmas, "seconds.")

    return chancelleryBlocks


# TODO: preprocessing() gibt aktuell chancelleryBlocks zurück. Ist das noch up to date oder sollte chancelleryHTMLtexts zurückgegeben und im Verlaufe des Programms dann mit
#  mainData statt chancelleryHTMLtexts weitergearbeitet werden?

def read_file_from_disk(filename, var_type):
    output = ""
    if var_type == "":
        with open(filename, 'r', encoding='utf-8') as fileToLoad:
            # Reading the JSON coded sequence from file
            output = json.load(fileToLoad)
    elif var_type == "DictWithTuple":
        with open(filename, 'rb') as fileToLoad:
            output = pickle.load(fileToLoad)
    return output


def loadModel():
    startLoadingModel = time.time()
    now = datetime.now()
    currentTime = now.strftime("%H:%M:%S")
    print("Loading model...\nCurrent time:", currentTime)
    possibleModelsToLoad = ["dewiki_20180420_100d.pkl.bz2", "dewiki_20180420_100d.txt.bz2", "dewiki_20180420_300d.txt.bz2",
                            "dewiki_20180420_100d.txt.bz2_loaded"]
    modelToLoad = 3
    global model
    if modelToLoad > 2:
        model = gensim.models.KeyedVectors.load(r'B:/Python-Projekte/Masterarbeit/Models/' + possibleModelsToLoad[modelToLoad], mmap='r')

    else:
        with Parallel(n_jobs=-1):
            model = gensim.models.KeyedVectors.load_word2vec_format(r'B:/Python-Projekte/Masterarbeit/' + possibleModelsToLoad[modelToLoad], binary=False,
                                                                    encoding='unicode escape')  # ,  workers=parallel)
            model.save(possibleModelsToLoad[modelToLoad] + "_loaded")
    model.fill_norms()

    timeSinceLoadingModel = round((time.time() - startLoadingModel) / 60, 3)
    print("Finished loading model. Time elapsed:", timeSinceLoadingModel, "minutes.")
    # winsound.PlaySound('SystemAsterisk.wav', winsound.SND_FILENAME)


startPreprocessing = time.time()


def linguistic_experiments(chancelleryHTMLtexts, chancelleriesWordDensities, lemmaCountsPerChancellery, chancelleriesPosTagCounts, chancelleriesSyntacticDependencies,
                           chancelleriesSentences):
    print("\n\nLINGUISTIC EXPERIMENTS:")
    #####################################
    #####################################
    #           Word density            #
    #####################################
    #####################################

    print(f"Length of chancelleriesWordDensities: {len(chancelleriesWordDensities)}")
    chancelleriesWordDensitesAverage = round(sum(chancelleriesWordDensities) / len(chancelleriesWordDensities), 2)
    chancelleriesWordDensities = {}
    chancelleriesWithZeroWordDensity = 0
    print("Average of word density over all chancelleries: {overallAverage}\n".format(overallAverage=chancelleriesWordDensitesAverage))
    for i, chancelleryBlock in enumerate(chancelleryHTMLtexts):
        currentChancelleryName = chancelleryBlock[0]
        warning = ""
        warningParameter = 50
        chancelleryWordDensity = round(chancelleryBlock[3]["averageWordDensity"], 2)
        chancelleryPercentageToAverage = round(((chancelleryWordDensity / chancelleriesWordDensitesAverage) * 100), 0)
        # for all chancelleries that have an average word density lower or higher than the warning parameter in percent compared to all other chancelleries
        # and if chancellery has a word density above 0 (0 would mean previously eliminated sentences due to various problems like multiple consecutive uppercase words
        # that result in no sentences left to be counted for a word density average
        if abs(chancelleryPercentageToAverage - 100) > warningParameter and chancelleryWordDensity > 0:
            warning = " !!!"
            print("\n\nWarning for chancellery __{chancellery}__!".format(chancellery=currentChancelleryName))
            chancellerySentences = chancelleriesSentences[currentChancelleryName]
            sentencesSplittedSorted = sorted(chancellerySentences, key=len)
            # print("length of splitted sorted sentences:", len(sentencesSplittedSorted))
            for k, sentence in enumerate(sentencesSplittedSorted):
                if k == (len(sentencesSplittedSorted) - 3):
                    print("Here come the 3 longest sentences:")
                if k > (len(sentencesSplittedSorted) - 3):  # k < 3 or
                    print("Sentence length:", len(sentence), "| sentence:", sentence)
                if k == 0:
                    print("Here come the 3 shortest sentences:")
                if k < 3:  # k < 3 or
                    print("Number,", k, "| Sentence length:", len(sentence), "| sentence:", sentence)
        elif chancelleryWordDensity == 0:
            chancelleriesWithZeroWordDensity += 1
        featureExpression = ""
        featureExpressionNumber = 0
        for feature in chancelleryBlock[2]:
            if feature[0] == "F23":
                featureExpression = feature[1]
                if featureExpression[1]:
                    featureExpressionNumber = featureExpression[-1]
                    # print(fr"Feature: {feature} | feature[1]: {feature[1]} |featureExpression[-1]: {featureExpression[-1]}")
                else:
                    print("Couldn't find a feature expression of feature F23 for chancellery", currentChancelleryName)
        # featureExpression = [featureComplexity for n, featureComplexity in enumerate(chancelleryBlock[2]) if n == "F23"]
        chancelleriesWordDensities[currentChancelleryName] = [chancelleryWordDensity, chancelleryPercentageToAverage, warning, featureExpressionNumber]

    dataframe = pd.DataFrame(chancelleriesWordDensities).transpose()
    # dataframe.rename({'0': "Chancellery's word density", '1': "Chancellery compared to average"}, axis=0)
    dataframe.columns = ["|C.'s word density", "|C. comp. to average in %", "|Warning", "Annotation"]
    pd.options.display.max_columns = 5
    pd.options.display.max_colwidth = 45
    pd.options.display.expand_frame_repr = False
    query = "!!!"
    warningOutput = dataframe[dataframe['|Warning'].str.contains(query)]
    if not warningOutput.empty:
        print(warningOutput)
    else:
        print("There is no problematic word density for any chancellery left. There is", chancelleriesWithZeroWordDensity,
              "of all chancelleries with an average word density of 0 which means every sentence had to be removed in the preprocessing.")

    print("The following dataframe contains information about the average word density of each chancellery:")
    print(dataframe)

    print("\nThe following list is the chancelleries' word densities compared to the annotation")
    chancelleriesWordDensitiesSorted = sorted(chancelleriesWordDensities.items(), key=lambda x: x[1][1])
    chancelleriesWordDensitiesSortedDict = {}

    wordDensityCategoryLow = [0, 15]
    wordDensityCategoryMedium = [15.1, 20]
    wordDensityCategoryHigh = [20.1, 30]

    wordDensityAccurateAnnotationsStrict = 0
    wordDensityAccurateAnnotationsLowerAndMediumValues = 0
    wordDensityAccurateAnnotationsMediumAndHighValues = 0

    for i, chancelleryBlock in enumerate(chancelleriesWordDensitiesSorted):
        # print(chancelleryBlock)
        chancelleryName = chancelleryBlock[0]
        chancelleryData = chancelleryBlock[1]
        chancelleryWordDensity = chancelleryData[0]
        chancelleryDensityPercentage = chancelleryData[1]
        featureExpressionNumber = int(chancelleryData[3])
        # print(f'{chancelleryName} | {chancelleryWordDensity} | {chancelleryDensityPercentage} | {featureExpressionNumber}')
        chancelleriesWordDensitiesSortedDict[chancelleryName] = [chancelleryWordDensity, chancelleryDensityPercentage, featureExpressionNumber]

        if wordDensityCategoryLow[0] < chancelleryWordDensity < wordDensityCategoryLow[1]:
            if featureExpressionNumber == 1:
                wordDensityAccurateAnnotationsStrict += 1
                wordDensityAccurateAnnotationsLowerAndMediumValues += 1
                wordDensityAccurateAnnotationsMediumAndHighValues += 1
            elif featureExpressionNumber == 2:
                wordDensityAccurateAnnotationsLowerAndMediumValues += 1
        elif wordDensityCategoryMedium[0] < chancelleryWordDensity < wordDensityCategoryMedium[1]:
            if featureExpressionNumber == 2:
                wordDensityAccurateAnnotationsStrict += 1
                wordDensityAccurateAnnotationsLowerAndMediumValues += 1
                wordDensityAccurateAnnotationsMediumAndHighValues += 1
            elif featureExpressionNumber == 1:
                wordDensityAccurateAnnotationsLowerAndMediumValues += 1
            elif featureExpressionNumber == 3:
                wordDensityAccurateAnnotationsMediumAndHighValues += 1
        elif wordDensityCategoryHigh[0] < chancelleryWordDensity < wordDensityCategoryHigh[1]:
            if featureExpressionNumber == 3:
                wordDensityAccurateAnnotationsStrict += 1
                wordDensityAccurateAnnotationsLowerAndMediumValues += 1
                wordDensityAccurateAnnotationsMediumAndHighValues += 1
            elif featureExpressionNumber == 2:
                wordDensityAccurateAnnotationsMediumAndHighValues += 1

    dataframeWithAnnotations = pd.DataFrame(chancelleriesWordDensitiesSortedDict).transpose()
    # dataframe.rename({'0': "Chancellery's word density", '1': "Chancellery compared to average"}, axis=0)
    dataframeWithAnnotations.columns = ["|C.'s word density", "| comp. to average in %", "Annotation"]
    pd.options.display.max_colwidth = 30
    print(dataframeWithAnnotations)

    wordDensityPerformanceStrict = wordDensityAccurateAnnotationsStrict / len(chancelleriesWordDensitiesSorted)
    wordDensityPerformanceLowerAndMediumValues = wordDensityAccurateAnnotationsLowerAndMediumValues / len(chancelleriesWordDensitiesSorted)
    wordDensityPerformanceMediumAndHighValues = wordDensityAccurateAnnotationsMediumAndHighValues / len(chancelleriesWordDensitiesSorted)
    print(f"\nReached a strict performance ratio of {round(wordDensityPerformanceStrict, 2)}")
    print(f"Reached a moderate performance ratio for both low and medium values of {round(wordDensityPerformanceLowerAndMediumValues, 2)}")
    print(f"Reached a moderate performance ratio for both medium and high values of {round(wordDensityPerformanceMediumAndHighValues, 2)}")

    #############################################
    #############################################
    ##              Word count                 ##
    #############################################
    #############################################

    # Sort the dictionary of a cumulated word count of all chancelleries
    sortedWordCountsCumulated = sorted(wordCountsCumulated.items(), key=lambda item: item[1], reverse=True)

    if sortedWordCountsCumulated:
        print("These are the 10 most common words among ALL chancelleries:")
        for i, word in enumerate(sortedWordCountsCumulated):
            if i < 10:
                print(i, ":", word)

    # Sort the dictionary of each chancellery's word count
    sortedWordCountsPerChancellery = []
    sortedLemmaCountsPerChancellery = []

    empathyWordCounts = {}

    wordsListEmpathyVerbs = ["achten", "anerkennen", "begleiten", "begreifen", "beistehen", "beschützen", "erkennen", "fühlen", "helfen", "lieben", "nachfühlen", "respektieren",
                             "unterstützen", "verstehen", "verzeihen", "würdigen", "mitleiden", "solidarisieren", "menschlich sein", "Anteil nehmen", "teilen",
                             "begreifen", "durchschauen", "einsehen", "erkennen", "kapieren", "verstehen", "nachvollziehen",
                             "verstehen können", "zustimmen", "wissen", "mitfühlen", "einfühlen", "nachempfinden"]
    wordsListEmpathyAdjectives = ["achtsam", "barmherzig", "begütigend", "einfühlsam", "ermutigend", "fürsorglich", "hilfsbereit", "liebevoll", "nachsichtig", "rücksichtsvoll",
                                  "sensibel", "verständnisvoll", "verzeihend", "mitleidig", "solidarisch", "menschlich", "anteilnehmend", "teilend", "unterstützend", "betroffen"]
    wordListEmpathy = wordsListEmpathyVerbs + wordsListEmpathyAdjectives

    # Read all chancellery specific lemmaCounts, transfer them into a dict, then a list and sort the list for each chancellery
    # lemmaCountsPerChancellery is a dictionary with tuple(lemma, posTag) as keys and frequencies in text as values
    for currentChancelleryName, chancelleryWordCounts in lemmaCountsPerChancellery.items():
        # print("key:", key, "| value:", value)
        currentChancelleryLemmas = {}
        empathyWordCounts[currentChancelleryName] = 0
        # Iterating over the lemma & word count dictionary
        for lemmaPosGroup, lemmaCount in chancelleryWordCounts.items():
            lemma = lemmaPosGroup[0]
            posTag = lemmaPosGroup[1]
            lemmaCount = chancelleryWordCounts[lemmaPosGroup]
            # Counting the occurences of empathic words
            if lemma in wordListEmpathy:
                empathyWordCounts[currentChancelleryName] += lemmaCount
            # Splitting each lemmaCountGroup in its lemma count & pos-tag and saving it to the dictionary
            currentChancelleryLemmas[(lemma, posTag)] = lemmaCount

        sortedLemmaCountsPerChancellery.append([currentChancelleryName, sorted(currentChancelleryLemmas.items(), key=lambda item: item[1], reverse=True)])

    def print_sorted_list(list_name, amount_of_lines):
        print("These are the,", amount_of_lines, "most common words for each chancellery:")
        for m, chancelleryGroup in enumerate(list_name):
            chancelleryName = chancelleryGroup[0]
            currentChancelleryLemmaCountList = chancelleryGroup[1]
            print("\n|", chancelleryName, "|")
            for p, currentLemma in enumerate(currentChancelleryLemmaCountList):
                if p < amount_of_lines:
                    print(p, currentLemma)

    print_sorted_list(sortedLemmaCountsPerChancellery, 5)

    ###################################
    # Comparison with frequency lists #
    ###################################

    # Loading the frequency list as a dataframe from file
    dataframeDerewo = pd.read_csv(r'B:\Python-Projekte\Masterarbeit\DeReKo-2014-II-MainArchive-STT.100000.freq', sep='\t', header=None, names=['word', 'lemma', 'pos', 'freq'])

    # Reducing the frequency list (word is not needed)
    dataframeDerewoReduced = dataframeDerewo[['lemma', 'pos', 'freq']]

    # Converting the dataframe to a list
    freqListDerewo = dataframeDerewoReduced.values.tolist()

    # Creating a dictionary from the list
    freqDictDerewo = {}
    for lemma, pos, freq in freqListDerewo:
        freqDictDerewo[lemma] = [pos, freq]

    print(f"Length of freqDictDerewo: {len(freqDictDerewo)}")

    print(f"Random 5 words of freqDictDerewo Gansel:{random.sample(list(freqDictDerewo.keys()), 5)}")

    chancelleryAverageDifferenceToFreqList = {}
    chancelleryOverallLemmaCount = {}

    # Iterating through the list of chancellery lemma counts
    # sortedLemmaCountsPerChancellery represented as chancelleryName : sorted list of [(lemma, posTag) : lemmaCount], [...], ...
    for i in range(len(sortedLemmaCountsPerChancellery)):
        chancellery = sortedLemmaCountsPerChancellery[i][0]
        lemmasGroup = sortedLemmaCountsPerChancellery[i][1]

        overallLemmaCount = 0

        # Creating a variable to store the total difference between the chancellery and the frequency list
        totalDiff = 0

        # Iterating over the lemmas of each chancellery
        for lemmaGroup in lemmasGroup:
            overallLemmaCount += 1
            lemma = lemmaGroup[0][0]
            posTag = lemmaGroup[0][1]
            lemmaCount = lemmaGroup[1]
            # If the chancellery's lemma is also in the dictionary of Derewo Frequencies
            if lemma in freqDictDerewo:
                freqDictLemmaPos = freqDictDerewo[lemma][0]
                freqDictLemmaCount = freqDictDerewo[lemma][1]
                if posTag != "" and posTag == freqDictLemmaPos:
                    diff = abs(lemmaCount - freqDictLemmaCount)
                    totalDiff += diff
            else:
                # If the current lemma is not in the frequency list, it's considered as an exception word
                pass
        # Divide the total difference by the total number of words in the law firm's website
        # to get a ratio of how different the word choice is from the general word frequency list
        averageDiff = totalDiff / len(sortedLemmaCountsPerChancellery)
        chancelleryAverageDifferenceToFreqList[chancellery] = round(averageDiff, 1)
        chancelleryOverallLemmaCount[chancellery] = overallLemmaCount
    # print(f"chancelleryAverageDifferenceToFreqList:\n{chancelleryAverageDifferenceToFreqList}")

    # Collecting all chancellery values if there's an annotation for complexity/ choice of words
    chancelleryComplexityAnnoations = {}
    complexityFalsePositives = 0
    complexityFalseNegatives = 0
    complexityTruePositives = 0
    complexityTrueNegatives = 0

    chancelleryAverageDifferenceToFreqListCumulated = [differenceValue for differenceValue in chancelleryAverageDifferenceToFreqList.values()]
    complexityPercentilesOld = np.percentile(chancelleryAverageDifferenceToFreqListCumulated, [33.33, 66.66])
    complexityPercentiles = np.percentile(chancelleryAverageDifferenceToFreqListCumulated, [50])
    print(f"chancelleryAverageDifferenceToFreqListCumulated percentiles: {complexityPercentiles}")

    for chancelleryBlock in chancelleryHTMLtexts:
        featureExpression = chancelleryBlock[2]
        chancelleryName = chancelleryBlock[0]
        for feature in featureExpression:
            if feature[0] == "F23":
                complexityAnnotation = int(feature[1][-1])
                chancelleryComplexityAnnoations[chancelleryName] = complexityAnnotation
                chancelleryAverageDiff = chancelleryAverageDifferenceToFreqList[chancelleryName]
                if complexityAnnotation > 1:
                    if chancelleryAverageDiff >= complexityPercentiles[0]:
                        complexityTruePositives += 1
                    if chancelleryAverageDiff < complexityPercentiles[0]:
                        complexityFalseNegatives += 1
                if complexityAnnotation == 1:
                    if chancelleryAverageDiff >= complexityPercentiles[0]:
                        complexityFalsePositives += 1
                    elif chancelleryAverageDiff < complexityPercentiles[0]:
                        complexityTrueNegatives += 1

    wordDensitiesCumulated = []
    for i, densityGroup in enumerate(chancelleriesWordDensities.items()):
        density = densityGroup[1][0]
        wordDensitiesCumulated.append(density)

    complexityAndWordDensityAccordance = 0
    complexityAndWordDensityValuesCount = 0
    wordDensityPercentiles = np.percentile(wordDensitiesCumulated, [50])
    print(f"wordDensityPercentiles: {wordDensityPercentiles}")

    for i, densityGroup in enumerate(chancelleriesWordDensities.items()):
        chancellery = densityGroup[0]
        density = densityGroup[1][0]
        for k, chancelleryBlock in enumerate(chancelleryAverageDifferenceToFreqList.items()):
            chancelleryName = chancelleryBlock[0]
            chancelleryDiffToFreqList = chancelleryBlock[1]
            if chancellery == chancelleryName:
                complexityAndWordDensityValuesCount += 1
                if density <= wordDensityPercentiles[0] and chancelleryDiffToFreqList <= complexityPercentiles[0]:
                    complexityAndWordDensityAccordance += 1
                elif density > wordDensityPercentiles[0] and chancelleryDiffToFreqList > complexityPercentiles[0]:
                    complexityAndWordDensityAccordance += 1
    print(f"complexityAndWordDensityAccordance: {complexityAndWordDensityAccordance} at {complexityAndWordDensityValuesCount} compared values.")

    print(f"chancellery| averageDifference | chancelleryComplexityAnnoations[chancellery] | chancelleriesWordDensities[chancellery]")
    for chancellery, averageDifference in chancelleryAverageDifferenceToFreqList.items():
        if chancellery in chancelleryComplexityAnnoations:
            print(
                f"{chancellery} | {round(averageDifference)} | {chancelleryComplexityAnnoations[chancellery]} | {chancelleriesWordDensities[chancellery][0]}")  # {chancelleryOverallLemmaCount[chancellery]}")

    complexityAccuracy = (complexityTruePositives + complexityTrueNegatives) / (
            complexityTruePositives + complexityTrueNegatives + complexityFalsePositives + complexityFalseNegatives)

    complexityRecall = complexityTruePositives / (complexityTruePositives + complexityFalseNegatives)

    complexityPrecision = complexityTruePositives / (complexityTruePositives + complexityFalsePositives)

    print(f"Calculated the following complexity metrics: Accuracy: {complexityAccuracy} | Recall: {complexityRecall} | Precision: {complexityPrecision}")

    ################################################
    ################################################
    ##          Similarities to word lists        ##
    ################################################
    ################################################

    # Dictionary for storing the ratio of empathy words to all words for chancellery
    empathyRatios = {}

    # Calculating the ratio of empathy words to all words for chancellery
    for chancelleryName, empathyCount in empathyWordCounts.items():
        lemmaCountTotal = 0
        for lemmaPosGroup, lemmaCount in lemmaCountsPerChancellery[chancelleryName].items():
            lemmaCountTotal += lemmaCount
        empathyRatios[chancelleryName] = empathyCount / lemmaCountTotal
    empathyRatiosSorted = sorted(empathyRatios.items(), key=lambda x: x[1])
    empathyRatiosSortedWithAnnotation = []
    empathyCorrectlyRecognizedCount = 0
    empathyCorrectlyRecognizedStrict = 0
    empathyDetectionNegatives = 0
    empathyTrueAnnotations = 0
    empathyRecognizedCount = 0  # The count of all cases where empathy was recognized for a chancellery (where the value is > 0)
    chancelleriesWithRecognizedEmpathy = {}
    chancelleriesWithEmpathyAnnotation = 0
    previousChancelleryName = ""

    print("\nThese are the empathy ratios of each chancellery with the annotated value:")
    # Assessing & summing up the correct annotations and printing out the empathy ratio of each chancellery
    for chancellery, chancelleryEmpathyRatio in empathyRatiosSorted:
        empathyAnnotation = 0
        for chancelleryBlock in chancelleryHTMLtexts:
            featureExpression = chancelleryBlock[2]
            if featureExpression and chancellery == chancelleryBlock[0]:
                # print("Feature expression:", featureExpression)
                # print("First feature:", featureExpression[0])
                for feature in featureExpression:
                    if feature[0] == "F8":
                        if chancellery != previousChancelleryName:
                            chancelleriesWithEmpathyAnnotation += 1
                            empathyAnnotation = feature[1][-1]
                            # Differentiating only between two groups:
                            # Chancelleries that have an empathy ratio above 0 and ones that don't
                            # If the chancellery has a ratio > 0 and the annotation of empathy is not 3 ("the opposite of empathy"), count it for the ratio
                            if int(empathyAnnotation) < 3:
                                empathyTrueAnnotations += 1
                                if chancelleryEmpathyRatio > 0:
                                    empathyCorrectlyRecognizedCount += 1
                            if chancelleryEmpathyRatio > 0:
                                chancelleriesWithRecognizedEmpathy[chancellery] = chancelleryEmpathyRatio
                                empathyRecognizedCount += 1
                            if int(empathyAnnotation) == 1 and chancelleryEmpathyRatio > 0:
                                empathyCorrectlyRecognizedStrict += 1
                            else:
                                empathyDetectionNegatives += 1
                            empathyRatiosSortedWithAnnotation.append([chancellery, chancelleryEmpathyRatio, empathyAnnotation])

                            print(f'{chancellery} | {chancelleryEmpathyRatio:.4f} | {empathyAnnotation}')  # chancellery, "|", ratio)
                            previousChancelleryName = chancellery

    empathyDetectionAccuracy = empathyCorrectlyRecognizedCount / chancelleriesWithEmpathyAnnotation
    empathyDetectionAccuracyStrict = empathyCorrectlyRecognizedStrict / chancelleriesWithEmpathyAnnotation
    empathyDetectionSensitivity = empathyCorrectlyRecognizedCount / empathyTrueAnnotations  # / (empathyAnnotationRatioStrict + empathyDetectionNegatives)
    empathyDetectionPrecision = empathyCorrectlyRecognizedCount / empathyRecognizedCount
    print(f"\nEmpathy detection accuracy of {round(empathyDetectionAccuracy, 2)}")
    print(f"Strict empathy detection accuracy of {round(empathyDetectionAccuracyStrict, 2)}")
    print(f"Empathy detection sensitivity of {round(empathyDetectionSensitivity, 2)}")
    print(f"Empathy detection precision of {round(empathyDetectionPrecision, 2)}")

    #####################################
    #####################################
    ##          PoS-tag count          ##
    #####################################
    #####################################

    print("\nLength of chancelleries pos tag count dict:", len(chancelleriesPosTagCounts.items()))
    print("Chancelleries' PoS-tag count:")

    chancelleriesAdjectivesCount = {}

    for chancellery, posCountBlock in chancelleriesPosTagCounts.items():
        posTagCountsTotal = 0
        for posTag, posTagCount in posCountBlock.items():
            posTagCountsTotal += posTagCount
        for posTag, posTagCount in posCountBlock.items():
            posTagRatio = round((posTagCount / posTagCountsTotal) * 100, 3)
            chancelleriesPosTagCounts[chancellery][posTag] = [posTagCount, posTagRatio]
            if posTag == "ADJ":
                chancelleriesAdjectivesCount[chancellery] = [posTagCount, posTagRatio]
            if chancellery == "gansel":
                print(f"posTag: {posTag} | posTagCount: {posTagCount} | posTagCountsTotal: {posTagCountsTotal} | posTagRatio: {posTagRatio} in percent: {posTagRatio} ")

    dataframePosTagCount = pd.DataFrame(chancelleriesPosTagCounts).transpose()
    # dataframe.rename({'0': "Chancellery's word density", '1': "Chancellery compared to average"}, axis=0)
    # dataframePosTagCount.columns = ["|C.'s word density", "| comp. to average in %", "Annotation"]
    pd.options.display.max_colwidth = 30
    pd.options.display.max_columns = 20
    print(dataframePosTagCount)

    adjectiveRatioAccurateAnnotationsStrict = 0
    adjectiveRatioAndEmpathyAnnotationsLowerAndMediumValues = 0
    adjectiveRatioAccurateAnnotationsMediumAndHighValues = 0

    sortedAdjectiveCountsPerChancellery = sorted(chancelleriesAdjectivesCount.items(), key=lambda item: item[1][1], reverse=True)

    chancelleriesAdjectiveCountRatioCumulated = []

    # Collecting all adjective counts per chancellery together
    print("Adjective ratio to empathy comparison")
    for i, chancelleryBlock in enumerate(sortedAdjectiveCountsPerChancellery):
        countGroup = chancelleryBlock[1]
        chancelleryAdjectiveRatio = countGroup[1]
        chancelleriesAdjectiveCountRatioCumulated.append(chancelleryAdjectiveRatio)
        for chancellery, chancelleryAdjectiveRatio in empathyRatiosSorted:
            if chancellery == chancelleryBlock[0]:
                print(f" {chancellery} | adjectiveRatio: {chancelleryAdjectiveRatio:.2f} | chancelleryEmpathyData: {chancelleryAdjectiveRatio:.4f} ")

    # Calculating the three quantiles for the three adjective ratio groups
    q1 = np.quantile(chancelleriesAdjectiveCountRatioCumulated, 0.25)
    q2 = np.quantile(chancelleriesAdjectiveCountRatioCumulated, 0.5)
    q3 = np.quantile(chancelleriesAdjectiveCountRatioCumulated, 0.75)

    AdjectivePercentiles = np.percentile(chancelleriesAdjectiveCountRatioCumulated, [33.33, 66.66])

    print(f"\nCalculated the following three quantiles for adjective groups: {q1}, {q2} and {q3}")

    chancelleriesEmpathyAndHighAdjectiveRatio = 0
    chancelleriesWithHighAdjectiveRatio = {}
    # Summing up how many chancelleries are inside the list of
    chancelleriesAccordanceEmpathyAndHighAdjectiveRatio = 0
    chancelleriesWithHigherAdjectiveRatioAndCorrectlyRecognizedEmpathy = 0
    chancelleriesWithHigherAdjectiveRatioAndRecognizedEmpathy = 0
    chancelleriesWithHigherAdjectiveRatioAndActualAnnotatedEmpathy = 0

    print("\nSorted adjective counts by ratio:")
    for i, chancelleryBlock in enumerate(sortedAdjectiveCountsPerChancellery):
        chancellery = chancelleryBlock[0]
        countGroup = chancelleryBlock[1]
        count = countGroup[0]
        chancelleryAdjectiveRatio = countGroup[1]
        empathyAnnotation = 0
        empathyRatio = 0
        for p, chancelleryBlockEmpathy in enumerate(empathyRatiosSortedWithAnnotation):
            if chancelleryBlockEmpathy[0] == chancellery:
                empathyAnnotation = int(chancelleryBlockEmpathy[2])
                empathyRatio = int(chancelleryBlockEmpathy[2])

        if chancelleryAdjectiveRatio <= AdjectivePercentiles[0]:
            if empathyAnnotation == 1:
                adjectiveRatioAccurateAnnotationsStrict += 1
                # adjectiveRatioAndEmpathyAnnotationsLowerAndMediumValues += 1
                adjectiveRatioAccurateAnnotationsMediumAndHighValues += 1
            # elif empathyAnnotation == 2:
            # adjectiveRatioAndEmpathyAnnotationsLowerAndMediumValues += 1

        if AdjectivePercentiles[0] < chancelleryAdjectiveRatio <= AdjectivePercentiles[1]:
            if empathyAnnotation == 2:
                adjectiveRatioAccurateAnnotationsStrict += 1
                # adjectiveRatioAndEmpathyAnnotationsLowerAndMediumValues += 1
                adjectiveRatioAccurateAnnotationsMediumAndHighValues += 1
            # elif empathyAnnotation == 1:
            #     adjectiveRatioAndEmpathyAnnotationsLowerAndMediumValues += 1
            elif empathyAnnotation == 3:
                adjectiveRatioAccurateAnnotationsMediumAndHighValues += 1
        if AdjectivePercentiles[1] < chancelleryAdjectiveRatio:
            chancelleriesWithHighAdjectiveRatio[chancellery] = chancelleryAdjectiveRatio
            if empathyAnnotation == 3:
                adjectiveRatioAccurateAnnotationsStrict += 1
                adjectiveRatioAndEmpathyAnnotationsLowerAndMediumValues += 1
                adjectiveRatioAccurateAnnotationsMediumAndHighValues += 1
            if empathyAnnotation == 2:
                adjectiveRatioAccurateAnnotationsMediumAndHighValues += 1
                adjectiveRatioAndEmpathyAnnotationsLowerAndMediumValues += 1

        # If there's a higher adjective ratio, a recognized empathy ratio above 0
        if AdjectivePercentiles[1] < chancelleryAdjectiveRatio:
            if empathyRatio > 0:
                chancelleriesWithHigherAdjectiveRatioAndRecognizedEmpathy += 1
                # If there's also an empathy annotation lower 3 (no anti-empathy)
                if empathyAnnotation < 3:
                    chancelleriesWithHigherAdjectiveRatioAndCorrectlyRecognizedEmpathy += 1
            if empathyAnnotation < 3:
                chancelleriesWithHigherAdjectiveRatioAndActualAnnotatedEmpathy += 1

        print(f"{chancellery}: {count} | {chancelleryAdjectiveRatio} | {empathyAnnotation}")

    for chancellery, chancelleryAdjectiveRatio in chancelleriesWithHighAdjectiveRatio.items():
        for chancelleryName, chancelleryEmpathyRatio in chancelleriesWithRecognizedEmpathy.items():
            if chancellery == chancelleryName:
                chancelleriesEmpathyAndHighAdjectiveRatio += 1

    # TODO: Prüfen welche Berechnung jetzt die korrekte ist, dann Klassifikator wieder einkommentieren und fortsetzen
    chancelleriesEmpathyAndHighAdjectiveRatioShare1 = chancelleriesEmpathyAndHighAdjectiveRatio / len(chancelleriesWithRecognizedEmpathy)
    chancelleriesEmpathyAndHighAdjectiveRatioShare2 = chancelleriesEmpathyAndHighAdjectiveRatio / len(chancelleriesWithHighAdjectiveRatio.items())
    chancelleriesEmpathyAndHighAdjectiveRatioShare3 = chancelleriesEmpathyAndHighAdjectiveRatio / chancelleriesWithEmpathyAnnotation
    adjectiveRatioAccuracyStrict = adjectiveRatioAccurateAnnotationsStrict / len(chancelleriesAdjectiveCountRatioCumulated)
    adjectiveRatioAccuracyLowerAndMediumValues = adjectiveRatioAndEmpathyAnnotationsLowerAndMediumValues / len(chancelleriesAdjectiveCountRatioCumulated)
    adjectiveRatioAccuracyMediumAndHighValues = adjectiveRatioAccurateAnnotationsMediumAndHighValues / len(chancelleriesAdjectiveCountRatioCumulated)
    # print(f"\nReached a strict adjective ratio accuracy of {round(adjectiveRatioAccuracyStrict * 100, 2)}")
    # print(f"Reached a moderate adjective ratio accuracy for both low and medium empathy values of {round(adjectiveRatioAccuracyLowerAndMediumValues * 100, 2)}")
    # print(f"Reached a moderate adjective ratio accuracy for both medium and high empathy values of {round(adjectiveRatioAccuracyMediumAndHighValues * 100, 2)}")
    print(
        f"\nThere are {chancelleriesEmpathyAndHighAdjectiveRatio} of all {len(chancelleriesWithRecognizedEmpathy)} chancelleries that have an empathy ratio above zero and an adjective ratio in the upper quantile. "
        f"That's {chancelleriesEmpathyAndHighAdjectiveRatioShare1 * 100:.2f}%!")
    print(f"There are {len(chancelleriesWithRecognizedEmpathy)} chancelleries with a recognized empathy.")

    adjectiveRatioSensitivity = chancelleriesWithHigherAdjectiveRatioAndCorrectlyRecognizedEmpathy / chancelleriesWithHigherAdjectiveRatioAndActualAnnotatedEmpathy  # / (empathyAnnotationRatioStrict + empathyDetectionNegatives)
    # adjectiveRatioSensitivity = adjectiveRatioAndEmpathyAnnotationsLowerAndMediumValues / empathyTrueAnnotations  # / (empathyAnnotationRatioStrict + empathyDetectionNegatives)
    # adjectiveRatioPrecision = adjectiveRatioAndEmpathyAnnotationsLowerAndMediumValues / len(chancelleriesWithHighAdjectiveRatio.items())
    adjectiveRatioPrecision = chancelleriesWithHigherAdjectiveRatioAndCorrectlyRecognizedEmpathy / chancelleriesWithHigherAdjectiveRatioAndRecognizedEmpathy
    print(f"Adjective detection sensitivity of {adjectiveRatioSensitivity:.2f}")
    print(f"Adjective detection precision of {adjectiveRatioPrecision:.2f}")
    print(
        f"{len(chancelleriesWithHighAdjectiveRatio.items())} chancelleries with high adjective ratio. {adjectiveRatioAndEmpathyAnnotationsLowerAndMediumValues} with high values "
        f"for empathy (annotation) & adjectives\n")

    ##########################################
    ##              Classifier              ##
    ##########################################

    from sklearn.svm import SVC
    chancelleriesTexts = []
    chancelleriesTextsDict = {}
    chancelleryNames = []
    chancelleryEmpathyLabels = {}
    chancelleriesNamesWithEmpathyAnnotation = []
    chancelleriesSentencesIfEmpathyLabels = {}

    # Saving the chancellery names and empathy labels, counting the occurences of the F8 feature annotation (empathy)
    chancelleriesWithF8Count = 0
    for k, chancelleryBlock in enumerate(chancelleryHTMLtexts):
        # Representation of chancelleryHTMLtexts: [currentChancelleryName, currentHTMLwordsList, currentFeatureExpressions, chancelleryCalculations]
        # Representation of currentFeatureExpressions (chancelleriesFeatureExpressionsWordsList): [[featureId, featureExpression], [featureId, ... ]]
        chancelleryName = chancelleryBlock[0]
        chancelleryFeatures = chancelleryBlock[2]
        # chancelleryTexts.append(chancelleryBlock[1])
        for featureGroup in chancelleryFeatures:
            if featureGroup[0] == "F8":
                # Checking that the current chancellery isn't doubled in the data
                if chancelleryName not in chancelleryNames:
                    chancelleryNames.append(chancelleryName)
                    chancelleriesWithF8Count += 1
                    chancelleryEmpathyLabels[chancelleryName] = featureGroup[1][-1]
                    chancelleriesNamesWithEmpathyAnnotation.append(chancelleryName)
                    chancelleriesSentencesIfEmpathyLabels[chancelleryName] = chancelleriesSentences[chancelleryName]
                    # print(f"{chancelleryName} has F8 annotation. That's number {chancelleriesWithF8Count}")
                    continue
    print(f"Lenth of chancelleriesWithF8Count: {chancelleriesWithF8Count}")
    print(f"chancelleryEmpathyLabels: {chancelleryEmpathyLabels}")

    chancelleriesSentencesAsStringsIfEmpathyAnnotation = []

    # Uniting the words of each text again for the classifier to train on it
    for chancelleryName, chancellerySentencesGroup in chancelleriesSentences.items():
        # print("ChancelleryName:", chancelleryName)
        # print("ChancellerySentenceGroup:", chancellerySentencesGroup)
        chancelleryText = ""
        chancellerySentencesAsStringsIfEmpathyAnnotation = []
        if len(chancellerySentencesGroup) > 0 and chancelleryName in chancelleriesNamesWithEmpathyAnnotation:
            for n, sentence in enumerate(chancellerySentencesGroup):
                if len(sentence) > 0:
                    # print("Sentence:", n, sentence)
                    chancellerySentence = ""
                    for p, word in enumerate(sentence):
                        if p > 0:
                            chancellerySentence = chancellerySentence + " " + word
                        else:
                            chancellerySentence = word
                    chancellerySentencesAsStringsIfEmpathyAnnotation.append(" ".join(sentence))
                    if n > 0:
                        chancelleryText = chancelleryText + ". " + chancellerySentence
                    else:
                        chancelleryText = chancellerySentence
        chancelleriesSentencesAsStringsIfEmpathyAnnotation.append(chancellerySentencesAsStringsIfEmpathyAnnotation)
        if chancelleryText != "":
            chancelleriesTexts.append(chancelleryText)
            chancelleriesTextsDict[chancelleryName] = chancelleryText

    print("Length of chancelleryTexts:", len(chancelleriesTexts))
    # print("First 100 chars of first text:\n", chancelleriesTexts[0][:100])
    # print(chancelleriesTexts)

    from gensim.models import Word2Vec
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
    from sklearn.model_selection import train_test_split

    # Splitting the dataset and initializing the variables with the respective dataset split
    trainingSetSize = 0.8

    datasetSize = int(len(chancelleriesTexts))
    trainingDataSplit = round(datasetSize * trainingSetSize)
    testDataSplit = round(datasetSize * trainingSetSize)

    datasetSentencesSize = int(len(chancelleriesSentences))
    trainingDataSentencesSplit = round(datasetSentencesSize * trainingSetSize)
    testDataSentencesSplit = round(datasetSentencesSize * trainingSetSize)

    datasetSplit = True

    trainingDataTexts = {}
    trainingDataLabels = {}
    testDataTexts = {}
    testDataLabels = {}
    trainingDataSentences = {}
    testDataSentences = {}

    # Splitting the dataset dictionary in chancellery names for training and test data
    chancelleryNames = list(chancelleriesSentences.keys())
    # random.shuffle(chancelleryNames)
    splitIndex = int(trainingSetSize * len(chancelleryNames))
    trainChancelleries = chancelleryNames[:splitIndex]
    testChancelleries = chancelleryNames[splitIndex:]

    trainingData = []
    testData = []
    trainingDataLabels = []
    testDataLabels = []

    # Filling the training data and test data set using the chancellery names of each split
    for chancellery, sentencesList in chancelleriesSentencesIfEmpathyLabels.items():
        if chancellery in trainChancelleries:
            trainingData.append(sentencesList)
            # trainingDataLabels.append(chancelleryEmpathyLabels[chancellery])
        elif chancellery in testChancelleries:
            testData.append(sentencesList)
            # testDataLabels.append(chancelleryEmpathyLabels[chancellery])

    print(f"Overall dataset was split in a training data size of {len(trainingData)} with {len(trainingDataLabels)} training labels")

    # for key, value in chancelleriesSentences.items():
    #     # Wähle zufällig eine Anzahl an Elementen aus der Liste aus, die dem Prozentsatz der Testdaten entspricht
    #     test_count = int(len(value) * testsetSize)
    #     test_indices = random.sample(range(len(value)), test_count)
    #     # Trenne die Liste in diejenigen Elemente, die als Testdaten verwendet werden, und diejenigen, die als Trainingsdaten verwendet werden
    #     test_elements = [value[i] for i in test_indices]
    #     train_elements = [elem for i, elem in enumerate(value) if i not in test_indices]
    #     # Füge die getrennten Elemente den entsprechenden Dictionaries hinzu
    #
    #     trainingDataSentences[key] = train_elements
    #     testDataSentences[key] = test_elements
    nbOfSentencesToBePrinted = 3
    print(f"Printing the first {nbOfSentencesToBePrinted} sentences of training Data")
    for text in trainingDataTexts:
        print(text.split(". ")[:nbOfSentencesToBePrinted])
        # print(text)

    # classifierModel = model
    modelType = 2

    if modelType == 2:
        # Creating the Word2Vec model
        # classifierModel = gensim.models.KeyedVectors.load_word2vec_format(r'B:/Python-Projekte/Masterarbeit/Models/dewiki_20180420_100d.txt.bz2', binary=False,
        #                                                                   encoding='unicode escape')
        print("Loading predefined model with word2vec")
        # classifierModel = Word2Vec.load(r'B:/Python-Projekte/Masterarbeit/Models/dewiki_20180420_100d.txt.bz2', encoding='unicode escape')
        # classifierModel = KeyedVectors.load_word2vec_format(r'B:/Python-Projekte/Masterarbeit/Models/dewiki_20180420_100d.txt.bz2')
        # try:
        #     classifierModel = gensim.models.KeyedVectors.load(r'B:/Python-Projekte/Masterarbeit/Models/dewiki_20180420_100d.txt.bz2_loaded')
        # except FileNotFoundError:
        #     print("Pre-loaded model file not found. Please run preprocessing first.")
        # lassifierModel = gensim.models.KeyedVectors.load_word2vec_format(r'B:/Python-Projekte/Masterarbeit/Models/dewiki_20180420_100d.txt.bz2_loaded', encoding="unicode escape",
        #                                                                  binary=False)
        # classifierModel = gensim.models.KeyedVectors.load(r'B:/Python-Projekte/Masterarbeit/Models/dewiki_20180420_100d.txt.bz2_loaded', mmap='r')
        # wikiModelWordVectors = KeyedVectors.load_word2vec_format(datapath(r'B:/Python-Projekte/Masterarbeit/Models/dewiki_20180420_100d.txt.bz2'), binary=False, encoding='UTF-8')
        # wikiModelWordVectors.save('B:/Python-Projekte/Masterarbeit/Models/dewiki_20180420_100d.txt.bz2_loaded.vectors')  # , binary=False)
        wikiModelWordVectors = KeyedVectors.load('B:/Python-Projekte/Masterarbeit/Models/dewiki_20180420_100d.txt.bz2_loaded.vectors', mmap='r')
        print("Done loading classifier model. Beginning to train the model on the chancelley texts")
        # classifierModel.train(trainingDataTexts, total_examples=len(trainingDataTexts), epochs=10)
        print("Training the model on the chancellery texts done. Beginning to extract the word vectors of the model.")

        # wordVectors = classifierModel.wv.vocab  # vs. classifierModel.wv

        print("Done extracting the word vectors. Beginning to filter the vectors for all words of the data set")

        # Extracting the empathy vectors from the model
        empathyVectorDict = {}
        empathyVectorList = []
        for empathyWord in wordListEmpathy:
            if empathyWord in wikiModelWordVectors:
                empathyVectorDict[empathyWord] = wikiModelWordVectors[empathyWord]
                empathyVectorList.append(wikiModelWordVectors[empathyWord])

        """
         eine andere, vermutlich bessere Option ist, eine explizite Liste von Wörtern zu erstellen, die indikativ für Empathie sind. 
         
         Für diese ermitteln Sie die Vektoren aus ihrem vortrainierten Modell und packen sie in eine Liste, die Empathievektorliste.
         
         Für eine zu klassifizierende Seite berechnen Sie dann für jedes Wort die durchschnittliche Cosinus-Distanz zu den Wortvektoren in der Empathievektorliste.
          
         Das gibt eine Liste von Empathiedistanzen. Anschließend können Sie entweder das Minimum über alle 
         ermittelten Distanzen oder den Durchschnitt oder beides als Features für den Klassifizierer verwenden.
        """

        trainingDataVectors = []
        testDataVectors = []
        empathyDistancesTrainingData = {}
        empathyDistancesTestData = {}
        minimumEmpathyDistancesTrainingData = []
        minimumEmpathyDistancesTestData = []
        averageEmpathyDistancesTrainingData = []
        averageEmpathyDistancesTestData = []

        # for chancellery, text in trainingDataSentences.items():
        #     textVectors = []
        #     for sentences in text:
        #         for word in sentences:
        #             wordVector = wikiModelWordVectors[word]
        #             if word in wikiModelWordVectors:
        #                 trainingDataVectors.append(wordVector)
        #                 textVectors.append(wordVector)
        #     cosineDistancesToEmpathyVectors = cosine_distances(textVectors, empathyVectorList)
        #     averageCosineDistance = cosineDistancesToEmpathyVectors.mean()
        #     empathyDistancesTrainingData[chancellery] = averageCosineDistance
        #     minimumEmpathyDistancesTrainingData[chancellery] = np.min(averageCosineDistance)
        #     averageEmpathyDistancesTrainingData[chancellery] = np.mean(averageCosineDistance)

        # # Extracting the vectors for the training data texts
        # for chancellery, lemmaCountGroups in lemmaCountsPerChancellery.items():
        #     textVectors = []
        #     for lemmaGroup, lemmaCount in lemmaCountGroups.items():
        #         lemma, posTag = lemmaGroup
        #         if lemma in wikiModelWordVectors:
        #             wordVector = wikiModelWordVectors[lemma]
        #             trainingDataVectors.append(wordVector)
        #             textVectors.append(wordVector)
        #     cosineDistancesToEmpathyVectors = cosine_distances(textVectors, empathyVectorList)
        #     averageCosineDistance = cosineDistancesToEmpathyVectors.mean()
        #     empathyDistancesTrainingData[chancellery] = averageCosineDistance
        #     minimumEmpathyDistancesTrainingData.append(np.min(averageCosineDistance))
        #     averageEmpathyDistancesTrainingData.append(np.mean(averageCosineDistance))
        #
        # # Extracting the vectors for the test data texts
        # for chancellery, lemmaCountGroups in lemmaCountsPerChancellery.items():
        #     textVectors = []
        #     for lemmaGroup, lemmaCount in lemmaCountGroups.items():
        #         lemma, posTag = lemmaGroup
        #         if lemma in wikiModelWordVectors:
        #             wordVector = wikiModelWordVectors[lemma]
        #             testDataVectors.append(wordVector)
        #             textVectors.append(wordVector)
        #     cosineDistancesToEmpathyVectors = cosine_distances(textVectors, empathyVectorList)
        #     averageCosineDistance = cosineDistancesToEmpathyVectors.mean()
        #     empathyDistancesTestData[chancellery] = averageCosineDistance
        #     minimumEmpathyDistancesTestData.append(np.min(averageCosineDistance))
        #     averageEmpathyDistancesTestData.append(np.mean(averageCosineDistance))
        # TODO: PRüfen ob die Übergabe der averageCosineDistance hier so korrekt ist
        # TODO: Fehler mit inkonsistenten Inputvariablen beim Fitten des Klassifikators beheben &

        print(f"Length of trainChancelleries: {len(trainChancelleries)}")
        print(f"Length of chancelleriesSentencesIfEmpathyLabels: {len(chancelleriesSentencesIfEmpathyLabels)}")
        chancelleriesTextsIfInTrainingData = []
        for chancellery, sentencesList in chancelleriesSentencesIfEmpathyLabels.items():
            if chancellery in trainChancelleries:
                trainingData.append(sentencesList)
                trainingDataLabels.append(chancelleryEmpathyLabels[chancellery])
                chancelleriesTextsIfInTrainingData.append(chancelleriesTextsDict[chancellery])
                textVectors = []
                for lemmaGroup, lemmaCount in lemmaCountsPerChancellery[chancellery].items():
                    lemma, posTag = lemmaGroup
                    if lemma in wikiModelWordVectors:
                        wordVector = wikiModelWordVectors[lemma]
                        trainingDataVectors.append(wordVector)
                        textVectors.append(wordVector)
                cosineDistancesToEmpathyVectors = cosine_distances(textVectors, empathyVectorList)
                averageCosineDistance = cosineDistancesToEmpathyVectors.mean()
                # print(f"For training data calculated an averageCosineDistance of {averageCosineDistance} ")
                empathyDistancesTrainingData[chancellery] = averageCosineDistance
                averageEmpathyDistancesTrainingData.append(averageCosineDistance)
                minimumEmpathyDistancesTrainingData.append(np.min(averageCosineDistance))
            elif chancellery in testChancelleries:
                testData.append(sentencesList)
                testDataLabels.append(chancelleryEmpathyLabels[chancellery])
                textVectors = []
                for lemmaGroup, lemmaCount in lemmaCountsPerChancellery[chancellery].items():
                    lemma, posTag = lemmaGroup
                    if lemma in wikiModelWordVectors:
                        wordVector = wikiModelWordVectors[lemma]
                        trainingDataVectors.append(wordVector)
                        textVectors.append(wordVector)
                cosineDistancesToEmpathyVectors = cosine_distances(textVectors, empathyVectorList)
                averageCosineDistance = cosineDistancesToEmpathyVectors.mean()
                # print(f"For test data calculated an averageCosineDistance of {averageCosineDistance} ")
                empathyDistancesTrainingData[chancellery] = averageCosineDistance
                averageEmpathyDistancesTestData.append(averageCosineDistance)
                minimumEmpathyDistancesTestData.append(np.mean(averageCosineDistance))
        print(
            f"Length of minimumEmpathyDistancesTrainingData: {len(minimumEmpathyDistancesTrainingData)} | Length of averageEmpathyDistancesTrainingData: {len(averageEmpathyDistancesTrainingData)}")
        trainingFeatures = np.column_stack(([minimumEmpathyDistancesTrainingData, averageEmpathyDistancesTrainingData]))
        testFeatures = np.column_stack(([minimumEmpathyDistancesTestData, averageEmpathyDistancesTestData]))

        # Initializing the TfidfVectorizer and calculating the weights
        tfidf = TfidfVectorizer(analyzer='word', stop_words=stopwords.words('german'))
        tfidfVectors = tfidf.fit_transform(chancelleriesTextsIfInTrainingData)

        tfidf = TfidfVectorizer()
        tfidfVectors = tfidf.fit_transform(chancelleriesTextsIfInTrainingData)
        tfidfWeights = {lemma: tfidfVectors[0, tfidf.vocabulary_[lemma]] for lemma in tfidf.vocabulary_}

        # # berechne die gewichteten Wortvektoren
        # weightedWordVectors = []
        # for chancellery, sentencesList in chancelleriesSentencesIfEmpathyLabels.items():
        #     if chancellery in trainChancelleries:
        #         for i, lemmaGroup in enumerate(lemmaCountsPerChancellery[chancellery].items()):
        #             lemma, posTag = lemmaGroup
        #             if lemma in wikiModelWordVectors and lemma in tfidfWeights:
        #                 wordVector = wikiModelWordVectors[lemma]
        #                 tfidfWeight = tfidfWeights[lemma]
        #                 weightedWordVector = wordVector * tfidfWeight
        #                 weightedWordVectors.append(weightedWordVector)

        # berechne die Distanzen zu Empathie-Vektoren
        # cosineDistancesToEmpathyVectors = cosine_distances(weightedWordVectors, empathyVectorList)
        # averageCosineDistance = cosineDistancesToEmpathyVectors.mean()
        # trainingFeatures = np.column_stack(([minimumEmpathyDistancesTrainingData, averageEmpathyDistancesTrainingData]))
        # testFeatures = np.column_stack(([minimumEmpathyDistancesTestData, averageEmpathyDistancesTestData]))

        print(f"LENGTH OF chancelleriesSentencesAsStringsIfEmpathyAnnotation: {len(chancelleriesSentencesAsStringsIfEmpathyAnnotation)}")
        print(f"First 3 Sentences: {chancelleriesSentencesAsStringsIfEmpathyAnnotation[0][:3]}")
        print(f"Length of chancelleriesTextsIfInTrainingData: {len(chancelleriesTextsIfInTrainingData)}")

        # indices = np.argsort(tfidfVectors.get_feature_names())
        # tfidfDict = dict(zip(tfidf.get_feature_names(), tfidf.idf_))

        print(f"length of tfidfVectors.toarray(): {len(tfidfVectors.toarray())}")
        print(f"{len(trainingFeatures)} trainingFeatures & {len(testFeatures)} testFeatures")
        # Combining the TfIdf vectors with the empathy vectors

        print(trainingFeatures.shape)
        print(tfidfVectors.toarray().shape)

        # combinedFeatures = np.hstack((trainingFeatures, tfidfVectors.toarray()))

        # Initializing a classifier
        classifier = SVC(kernel='linear', C=1, probability=True)  # , random_state=42)

        # Training the classifier with the training data vectors and labels
        classifier.fit(trainingFeatures, trainingDataLabels)

        # Making predictions on the test data vectors
        predictions = classifier.predict(testFeatures)

        # Evaluating the classifier's performance
        accuracy = accuracy_score(testDataLabels, predictions)
        recall = recall_score(testDataLabels, predictions, average='macro', zero_division=False)
        precision = precision_score(testDataLabels, predictions, average='weighted', zero_division=False)
        print(f"Metrics of model approach {modelType}: Accuracy of {accuracy} | Sensitivity of {recall} | precision of {precision}")

        scores = cross_val_score(classifier, trainingFeatures, trainingDataLabels, cv=5)
        print("Cross validation scores: {}".format(scores))
        print("Average score: {:.2f}".format(scores.mean()))

    elif modelType == 1:
        # Initializing the model
        classifierModel = Doc2Vec(vector_size=100, window=5, min_count=1, workers=4)

        # Building the vocabulary
        from gensim.models.doc2vec import TaggedDocument
        # documents = [TaggedDocument(trainingDataTexts.split(), [i]) for i, text in enumerate(trainingDataTexts)]
        # documents = [TaggedDocument([word for sentence in text for word in sentence.split()], [i]) for i, text in enumerate(trainingDataTexts)]
        # documents = [TaggedDocument(text, [i]) for i, text in enumerate(trainingDataTexts)]
        print("before preparing the documents")
        documents = [TaggedDocument([word for sentence in text for word in sentence.split()], [i]) for i, text in enumerate(trainingDataTexts)]
        print("after preparing the documents, before building the vocabulary")

        classifierModel.build_vocab(documents)

        print("After building the vocabulary, before training the model")

        # Training the model
        classifierModel.train(documents, total_examples=classifierModel.corpus_count, epochs=10)

        print("After training the model, bevore 2nd preprocessing the data")
        # 2nd Preprocessing of data to have it as words
        words = []
        for document in testDataTexts:
            for sentence in document:
                for word in sentence.split():
                    words.append(word)
        print("After 2nd preprocessing the data, before inferring the vectors")

        # Inferring the vectors
        vectors = []
        vector = classifierModel.infer_vector(words)
        vectors.append(vector)

        print(len(vectors))
        print(vectors)
        print(len(testDataLabels))

        # Using the vectors as input for the classifier
        classifierModel.fit(vectors, testDataLabels)

        # Making predictions
        predictions = classifierModel.predict(vectors)

        # Computing the accuracy
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(testDataLabels, predictions)

        # printing the accuracy
        print("Accuracy in Doc2Vec approach:", accuracy)

    elif modelType == 0:

        # Creating the Word2Vec model
        classifierModel = Word2Vec(trainingDataTexts, vector_size=100, window=5, min_count=1, workers=4)
        wordVectors = classifierModel.wv  # vs. classifierModel.wv

        # Extracting the vectors for the training data texts
        trainingDataVectors = []
        for chancelleryName, chancellerySentenceGroup in trainingDataSentences:
            textVectors = []
            for sentence in chancellerySentenceGroup:
                for word in sentence:
                    if word in wordVectors:
                        textVectors.append(wordVectors[word])
            avgVector = np.nanmean(textVectors)  # np.mean(textVectors)  # sum(text_vectors) / len(text_vectors)
            trainingDataVectors.append(avgVector)

        # Extracting the vectors for the test data texts
        testDataVectors = []
        for chancelleryName, chancellerySentenceGroup in testDataSentences:
            textVectors = []
            for sentence in chancellerySentenceGroup:
                for word in sentence:
                    if word in wordVectors:
                        textVectors.append(wordVectors[word])
            avgVector = np.nanmean(textVectors)  # np.mean(textVectors)
            testDataVectors.append(avgVector)

        print(f"Number of trainingDataVectors: {len(trainingDataVectors)} | number of trainingDataLabels: {len(trainingDataLabels)}")

        # Initializing a classifier
        classifier = SVC(kernel='linear', C=1, probability=True)  # , random_state=42)

        trainingDataVectors = np.array(trainingDataVectors).reshape(-1, 1)
        testDataVectors = np.array(testDataVectors).reshape(-1, 1)
        trainingDataVectors = np.nan_to_num(trainingDataVectors)
        testDataVectors = np.nan_to_num(testDataVectors)

        trainingDataVectors, trainingDataLabels = zip(*((v, l) for v, l in zip(trainingDataVectors, trainingDataLabels) if v is not None))
        testDataVectors, testDataLabels = zip(*((v, l) for v, l in zip(testDataVectors, testDataLabels) if v is not None))

        # Training the classifier with the training data vectors and labels
        classifier.fit(trainingDataVectors, trainingDataLabels)

        # Making predictions on the test data vectors
        predictions = classifier.predict(testDataVectors)

        # Evaluating the classifier's performance
        accuracy = accuracy_score(testDataLabels, predictions)
        recall = recall_score(testDataLabels, predictions, average='macro', zero_division=False)
        precision = precision_score(testDataLabels, predictions, average='weighted', zero_division=False)
        print(f"Metrics of model approach {modelType}: Accuracy of {accuracy}| Sensitivity of {recall}| precision of {precision}")
    elif modelType == -1:
        # Initializing the TfidfVectorizer and setting the  n-gram ranges (here: 2-Grams)
        vectorizer = TfidfVectorizer(ngram_range=(2, 2), stop_words=None)

        # Transforming the text contents in vectors with tf-idf-values
        trainingVectors = vectorizer.fit_transform(trainingDataTexts)

        # Initializing a support vector machine as classificator
        classifier = SVC(kernel='linear', C=1, probability=True, random_state=42)

        # Training the classifier with the tf-idf-vectors and the class labels
        classifier.fit(trainingVectors, trainingDataLabels)

        ############################## Applying the classifier onto new data (test data) ##################################
        # Transforming the test data in vectors with tf-idf values
        testVectors = vectorizer.transform(testDataTexts)

        # Using the classifier to classificate the new texts
        predictions = classifier.predict(testVectors)

        # Creating a dictionary with all texts of the test data set and the matching labels and predictions
        # results = {"labels": testDataLabels, "predictions": predictions}  # "text": testDataTexts,
        results = zip(testDataLabels, predictions)

        # Printing the predictions
        for label, prediction in results:
            print(f"{label} : {prediction}")

        # Evaluating the classifier's performance
        accuracy = accuracy_score(testDataLabels, predictions)
        recall = recall_score(testDataLabels, predictions, average='macro', zero_division=True)
        precision = precision_score(testDataLabels, predictions, average='weighted', zero_division=True)

        print(f"Metrics of model approach {modelType}: Accuracy of {accuracy} | Sensitivity of {recall} | precision of {precision}")

        conf_matrix = confusion_matrix(testDataLabels, predictions)
        accuracy = sum(conf_matrix.diagonal()) / sum(sum(conf_matrix))

        print(f"Calculation with confuison_matrix:")
        print(f"Metrics of model approach {modelType}: Accuracy of {accuracy}")
        print(f"Calculation with classification_report:")
        print(classification_report(testDataLabels, predictions))


readFilesFromDisk = None
readFilesFromDiskInput = input("Read files from disk? If not, enter 'n'. Else just press Enter")
if "n" in readFilesFromDiskInput.lower():
    readFilesFromDisk = False
else:
    readFilesFromDisk = True

if readFilesFromDisk:
    chancelleryHTMLtextsFromFile = read_file_from_disk('chancelleryHTMLtexts.txt', "")
    lemmaCountsPerChancelleryFromFileRaw = read_file_from_disk('lemmaCountsPerChancellery.txt', "tuple")
    # lemmaCountsPerChancelleryFromFile = {tuple(key.split(', ')): value for key, value in lemmaCountsPerChancelleryFromFileRaw.items()}
    lemmaCountsPerChancelleryFromFile = read_file_from_disk("lemmaCountsPerChancellery.txt", "DictWithTuple")
    chancelleriesWordDensities = read_file_from_disk('chancelleriesWordDensities.txt', "")
    chancelleriesPosTagCountsFromFile = read_file_from_disk('chancelleriesPosTagCounts.txt', "")
    chancelleriesSyntacticDependenciesFromFile = read_file_from_disk('chancelleriesSyntacticDependencies.txt', "")
    chancelleriesSentencesFromFile = read_file_from_disk('chancelleriesSentences.txt', "")

    linguistic_experiments(chancelleryHTMLtextsFromFile, chancelleriesWordDensities, lemmaCountsPerChancelleryFromFile, chancelleriesPosTagCountsFromFile,
                           chancelleriesSyntacticDependenciesFromFile, chancelleriesSentencesFromFile)

#     "Loading websitesTextsReformatted & chancelleryHTMLtexts from file."
#     with open('chancelleryHTMLtexts.txt', 'r', encoding='utf-8') as f:
#         # Reading the JSON coded sequence from file
#         chancelleryHTMLtextsFromDisk = json.load(f)
#
#     with open('currentChancelleryLemmaCount.txt', 'r', encoding='utf-8') as f:
#         # Reading the JSON coded sequence from file
#         lemmaCountsPerChancellery = json.load(f)
else:

    print("Starting to preprocess the annotated data...")
    mainData = preprocessing(
        r'B:/Python-Projekte/Masterarbeit/websitesTextsReformatted.txt')  # (r'C:/Users/Malte/Dropbox/Studium/Linguistik Master HHU/Masterarbeit/websitesTextsReformatted.txt')
    print("Finished preprocessing data. Time elapsed:", round(((time.time() - startPreprocessing) / 60), 2))
    print("Saving websitesTextsReformatted & chancelleryHTMLtexts & chancelleriesWordDensites to file.")

    # Opening the text file to write the list to it
    with open('chancelleryHTMLtexts.txt', 'w', encoding='utf-8') as f:
        # Transferring the dict in a JSON sequence and writing to file
        json.dump(chancelleryHTMLtexts, f, ensure_ascii=False)
        print("Saved a list of {} items to file '{}'.".format(len(chancelleryHTMLtexts), "chancelleryHTMLtexts.txt"))
    with open(r'B:/Python-Projekte/Masterarbeit/lemmaCountsPerChancellery.txt', 'wb') as f:
        # lemmaCountsPerChancelleryTransformed = {str(key): value for key, value in lemmaCountsPerChancellery.items()}
        pickle.dump(lemmaCountsPerChancellery, f)
        print("Saved a list of {} items to file '{}'.".format(len(lemmaCountsPerChancellery), "lemmaCountsPerChancellery.txt"))
    with open(r'B:/Python-Projekte/Masterarbeit/chancelleriesWordDensites.txt', 'w', encoding='utf-8') as f:
        json.dump(chancelleriesWordDensites, f, ensure_ascii=False)
        print("Saved a list of {} items to file '{}'.".format(len(chancelleriesWordDensites), "chancelleriesWordDensites.txt"))
    with open(r'B:/Python-Projekte/Masterarbeit/chancelleriesPosTagCounts.txt', 'w', encoding='utf-8') as f:
        json.dump(chancelleriesPosTagCounts, f, ensure_ascii=False)
        print("Saved a list of {} items to file '{}'.".format(len(chancelleriesPosTagCounts), "chancelleriesPosTagCounts.txt"))
    with open(r'B:/Python-Projekte/Masterarbeit/chancelleriesSyntacticDependencies.txt', 'w', encoding='utf-8') as f:
        json.dump(chancelleriesSyntacticDependencies, f, ensure_ascii=False)
        print("Saved a list of {} items to file '{}'.".format(len(chancelleriesSyntacticDependencies), "chancelleriesSyntacticDependencies.txt"))
    with open(r'B:/Python-Projekte/Masterarbeit/chancelleriesSentences.txt', 'w', encoding='utf-8') as f:
        json.dump(chancelleriesSentences, f, ensure_ascii=False)
        print("Saved a list of {} items to file '{}'.".format(len(chancelleriesSentences), "chancelleriesSentences.txt"))

    linguistic_experiments(chancelleryHTMLtexts, chancelleriesWordDensites, lemmaCountsPerChancellery, chancelleriesPosTagCounts, chancelleriesSyntacticDependencies,
                           chancelleriesSentences)

exit()


def printMainData(lineToPrint):
    print("Printing data line", lineToPrint,
          "\nFormat is: [[chancellery name, [featureId, featureName, text positions, [featureInfoActualData]]],chancelleryHTML]")
    print(mainData[lineToPrint])
