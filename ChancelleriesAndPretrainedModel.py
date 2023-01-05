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
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
from joblib import Parallel
from matplotlib import pyplot as plt
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import euclidean_distances
import json

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


def preprocessing(corpus_path):
    valuableFeatureIds = ["F8", "F22", "F23", "F24"]

    with open(datapath(corpus_path), 'r', encoding='utf-8') as f:  # former: encoding='unicode_escape'
        mainDataRaw = f.read()

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

        # For every line in the current chancellery block that is a list of lines (strings)
        for k in range(len(currentChancelleryBlock)):
            currentLine = currentChancelleryBlock[k]
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
                currentChancelleryHTMLclean = BeautifulSoup(currentChancelleryHTML, "html.parser").get_text()
                timeForParsingHtml += round((time.time() - startTimerProcessingHtmlLine), 2)

                currentWordDensities = []  # The list of all sentences' word density in the current document
                currentChancelleryHTMLclean = currentChancelleryHTMLclean.replace("\t", " ")
                currentChancelleryHTMLclean = currentChancelleryHTMLclean.replace("  ", "")
                currentSentences = currentChancelleryHTMLclean.split(".")  # TODO: Prüfen, dass hier die Entfernung des Leerzeichens nicht mehr Probleme gemacht hat.
                currentSentencesAfterThreshold = []
                minimumSentenceLength = 7

                # Keeping only those sentences that pass the threshold of minimum sentence length
                for sentence in currentSentences:
                    if len(sentence) >= minimumSentenceLength:
                        currentSentencesAfterThreshold.append(sentence)

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
                        if not sum(1 for c in word if c.isupper()) > 3 and len(word) > 1:
                            htmlDebris = ["\t", "\t\t", "\xa0", "[", "]", "'", ">>", "<<", "|"]
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
                            if "." in wordToAppend and wordToAppend.find(".") != len(wordToAppend) and wordToAppend.find(".") != 0:
                                wordSplitted = wordToAppend.split(".")
                            elif "/" in wordToAppend and wordToAppend.find("/") != len(wordToAppend) and wordToAppend.find("/") != 0:
                                wordSplitted = wordToAppend.split("/")
                            if len(wordSplitted) == 2:
                                if wordSplitted[0] != "":
                                    wordsCleaned.append(wordSplitted[0])
                                if wordSplitted[1] != "":
                                    wordsCleaned.append(wordSplitted[1])
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
                                # TODO: Hier alternativ mal nur dann das lemma weitergeben, wenn partOfSpeechTag "VERB" ist, ansonsten word weitergeben
                                #  (dann würden aber z.B. Nomen im Plural nicht weitergegeben)
                                lemmasWithPartsOfSpeech.append([lemma, partOfSpeechTag])

                            timeForNlpAndPosTagging += round((time.time() - startTimerNlpAndPosTagging), 2)

                            startTimerCountLemmas = time.time()
                            # Appending all cleaned words to the HTMLwordsList, the mainDataWordsList and the dictionary that counts the word count
                            # print("Appending all cleaned words to the HTMLwordsList, the mainDataWordsList and the dictionary that counts the word count")
                            for lemmaWithPos in lemmasWithPartsOfSpeech:
                                lemma = lemmaWithPos[0]
                                partOfSpeechTag = lemmaWithPos[1]
                                test = True
                                if lemma.lower().isalpha() and lemma.lower() not in germanStopWords:
                                    currentHTMLwordsList.append(lemma)  # [word.lower() for word in HTMLtokens if word.isalpha()]
                                    mainDataWordsList.append(lemma)
                                    if lemma in wordCountsCumulated:
                                        wordCountsCumulated[lemma] += 1
                                    else:
                                        wordCountsCumulated[lemma] = 1

                                    # Adding the cleaned word accompanied by its part of speech to the chancellery's word count dictionary
                                    # The style is: word : [[wordCountOfFirstOccurence, partOfSpeechOfFirstOccurence], [wordCountOfFirstOccurence, partOfSpeechOfSecondOcc...]
                                    # This way any ambiguity is saved if the part of speech of the current cleaned word is not the same as saved in the dict for this word

                                    # So if the lemma is already present in the dictionary,
                                    # check wether the pos-tag that is present for the lemma is the same as the current one
                                    if lemma in currentChancelleryLemmaCount:
                                        # If the currnet pos-tag is identical to the one in the dict for the lemma, just add one to the lemma count in the current list element
                                        currentPosTagCountList = currentChancelleryLemmaCount[lemma]
                                        # If there is more than one pos-tag group inside the list (that is the value for the lemma in the dict)
                                        if len(currentPosTagCountList) > 1:
                                            # For every list element of this list of pos-tags with lemma count
                                            for n, posTagLemmaGroup in enumerate(currentPosTagCountList):
                                                currentCount = posTagLemmaGroup[0]
                                                currentPosTag = posTagLemmaGroup[1]
                                                # If the current pos-tag of the upper lemmaWithPos list is identical
                                                # to the currently viewed pos tag of the pos tags list of the current lemma in the dict
                                                if currentPosTag == partOfSpeechTag:
                                                    listPartBeforeCurrentPosTag = currentChancelleryLemmaCount[lemma][:n]
                                                    listPartAfterCurrentPosTag = currentChancelleryLemmaCount[lemma][n + 1:]
                                                    currentChancelleryLemmaCount[lemma] = [listPartBeforeCurrentPosTag, [currentCount, currentPosTag],
                                                                                           listPartAfterCurrentPosTag]  # [currentChancelleryLemmaCount[lemma][0] + 1, partOfSpeechTag]
                                        else:
                                            currentLemmaCount = currentChancelleryLemmaCount[lemma][0][0]
                                            currentPosTag = currentChancelleryLemmaCount[lemma][0][1]
                                            currentChancelleryLemmaCount[lemma] = [[currentLemmaCount + 1, currentPosTag]]
                                    # Else add the combination lemmaCount, partOfSpeechTag als a new list element of a new list in the dict for the lemma
                                    else:
                                        currentChancelleryLemmaCount[lemma] = [[1, partOfSpeechTag]]

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
                                    if i == len(wordsCleaned) - 1 or not word[0].isupper():
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
        chancelleryLinguisticAssertions = {"averageWordDensity": averageWordDensity}

        lemmaCountsPerChancellery[currentChancelleryName] = currentChancelleryLemmaCount
        # currentAverageWordDensityCompared = sum(chancelleriesWordDensites) / len(chancelleriesWordDensites)
        chancelleryBlocks.append([newChancelleryBlock, currentHTMLwordsList])  # currentChancelleryHTMLclean])  # ,currentChancelleryHTML])
        chancelleriesFeatureExpressionsWordsList.append([currentChancelleryName, currentChancelleryWordsList])
        chancelleryHTMLtexts.append([currentChancelleryName, currentHTMLwordsList, currentFeatureExpressions, chancelleryLinguisticAssertions])  # currentChancelleryHTMLclean])

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

startPreprocessing = time.time()


def print_linguistic_assertions(chancelleryHTMLtexts, chancelleriesWordDensites, lemmaCountsPerChancellery):
    print("\n\nLINGUISTIC ASSERTIONS:")

    #################
    # Word density  #
    #################

    chancelleriesWordDensitesAverage = round(sum(chancelleriesWordDensites) / len(chancelleriesWordDensites), 2)
    chancelleriesWordDensities = {}
    chancelleriesWithZeroWordDensity = 0
    print("Average of word density over all chancelleries: {overallAverage}".format(overallAverage=chancelleriesWordDensitesAverage))
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
        chancelleriesWordDensities[currentChancelleryName] = [chancelleryWordDensity, chancelleryPercentageToAverage, warning]

    dataframe = pd.DataFrame(chancelleriesWordDensities).transpose()
    # dataframe.rename({'0': "Chancellery's word density", '1': "Chancellery compared to average"}, axis=0)
    dataframe.columns = ["|C.'s word density", "|C. comp. to average in %", "|Warning"]
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

    ##############
    # Word count #
    ##############

    # Sort the dictionary of a cumulated word count of all chancelleries
    sortedWordCountsCumulated = sorted(wordCountsCumulated.items(), key=lambda item: item[1], reverse=True)

    print("These are the 10 most common words among ALL chancelleries:")
    for i, word in enumerate(sortedWordCountsCumulated):
        if i < 10:
            print(i, ":", word)

    # Sort the dictionary of each chancellery's word count
    sortedWordCountsPerChancellery = []
    sortedLemmaCountsPerChancellery = []

    # Read all chancellery specific LemmaCounts, transfer them into a list and sort the list for each chancellery
    for currentChancelleryName, chancelleryWordCounts in lemmaCountsPerChancellery.items():
        # print("key:", key, "| value:", value)
        currentChancelleryLemmas = {}
        for lemma, lemmaCountGroups in chancelleryWordCounts.items():

            # Intercepting the case that there is ambiguity with more than one POS-tag for the current lemma and its lemma count
            # Until now, in this case this just gets printed but not processed furthermore
            if len(lemmaCountGroups) > 1:
                print("More than one POS-tag found for the current word:")
                for lemmaCountGroup in lemmaCountGroups:
                    print(lemmaCountGroup)
            # In case there is no ambiguity regarding thew current lemma, the only group of lemmaCountGroups (therefore [0]) is split in its lemma count & pos-tag
            elif len(lemmaCountGroups) == 1:
                lemmaCount = lemmaCountGroups[0][0]
                partOfSpeechTag = lemmaCountGroups[0][1]
                if lemma in currentChancelleryLemmas:
                    currentChancelleryLemmas[lemma] += [lemmaCount, partOfSpeechTag]
                else:
                    currentChancelleryLemmas[lemma] = lemmaCount
        sortedLemmaCountsPerChancellery.append([currentChancelleryName, sorted(currentChancelleryLemmas.items(), key=lambda item: item[1], reverse=True)])
        # sortedWordCountsPerChancellery.append([currentChancelleryName, sorted(chancelleryWordCounts.items(), key=lambda item: item[1], reverse=True)])

    def print_sorted_list(dict_name, amount_of_lines):
        print("These are the,", amount_of_lines, "most common words for each chancellery:")
        for m, chancelleryGroup in enumerate(dict_name):
            chancelleryName = chancelleryGroup[0]
            currentChancelleryWordCountList = chancelleryGroup[1]
            print("\n|", chancelleryName, "|")
            for n, currentWord in enumerate(currentChancelleryWordCountList):
                if n < amount_of_lines:
                    print(n, currentWord)

    print_sorted_list(sortedLemmaCountsPerChancellery, 10)


readFilesFromDisk = True


def read_file_from_disk(fileName):
    with open(fileName, 'r', encoding='utf-8') as fileToLoad:
        # Reading the JSON coded sequence from file
        output = json.load(fileToLoad)
    return output


if readFilesFromDisk:
    chancelleryHTMLtextsFromFile = read_file_from_disk('chancelleryHTMLtexts.txt')
    lemmaCountsPerChancelleryFromFile = read_file_from_disk('lemmaCountsPerChancellery.txt')
    chancelleriesWordDensites = read_file_from_disk('chancelleriesWordDensites.txt')
    print_linguistic_assertions(chancelleryHTMLtextsFromFile, chancelleriesWordDensites, lemmaCountsPerChancelleryFromFile)
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
    with open('chancelleryHTMLtexts.txt', 'w') as f:
        # Transferring the dict in a JSON sequence and writing to file
        json.dump(chancelleryHTMLtexts, f)
        print("Saved a list of {} items to file '{}'.".format(len(chancelleryHTMLtexts), "chancelleryHTMLtexts.txt"))
    with open(r'B:/Python-Projekte/Masterarbeit/lemmaCountsPerChancellery.txt', 'w') as f:
        json.dump(lemmaCountsPerChancellery, f)
        print("Saved a list of {} items to file '{}'.".format(len(lemmaCountsPerChancellery), "lemmaCountsPerChancellery.txt"))
    with open(r'B:/Python-Projekte/Masterarbeit/chancelleriesWordDensites.txt', 'w') as f:
        json.dump(chancelleriesWordDensites, f)
        print("Saved a list of {} items to file '{}'.".format(len(chancelleriesWordDensites), "chancelleriesWordDensites.txt"))

    print_linguistic_assertions(chancelleryHTMLtexts, chancelleriesWordDensites, lemmaCountsPerChancellery)

exit()


def printMainData(lineToPrint):
    print("Printing data line", lineToPrint,
          "\nFormat is: [[chancellery name, [featureId, featureName, text positions, [featureInfoActualData]]],chancelleryHTML]")
    print(mainData[lineToPrint])


printMainData(1)

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
    with Parallel(n_jobs=-1) as parallel:
        model = gensim.models.KeyedVectors.load_word2vec_format(r'B:/Python-Projekte/Masterarbeit/' + possibleModelsToLoad[modelToLoad], binary=False,
                                                                encoding='unicode escape', workers=4)  # ,  workers=parallel)

model.fill_norms()
# model.save(possibleModelsToLoad[modelToLoad] + "_loaded")

timeSinceLoadingModel = round((time.time() - startLoadingModel) / 60, 3)
print("Finished loading model. Time elapsed:", timeSinceLoadingModel, "minutes.")
# winsound.PlaySound('SystemAsterisk.wav', winsound.SND_FILENAME)
startComputingModelInfo = time.time()
print("Starting to compute some model info...")
# Check dimension of word vectors
print("Vector size:", model.vector_size)
print("The result of model.most_similar(positive=['koenig', 'frau'], negative=['mann']) is:\n",
      model.most_similar(positive=['koenig', 'frau'], negative=['mann']))
timeSinceComputingModelInfo = round((time.time() - startComputingModelInfo) / 60, 3)
print("Model info computed. Time Elapsed:", timeSinceComputingModelInfo)

# ModelWordVectors = model.wv

# Filter the list of vectors to include only those that Word2Vec has a vector for
vector_list = [model[word] for word in mainDataWordsList if word in model.key_to_index]
chancelleriesVectorList = []
# for i in range(len(mainData)):
#     chancelleryBlock = mainData[i]
#     chancelleryHTML = chancelleryBlock[-1]
#     for word in chancelleryBlock
# [model[word] for chancelleryBlock[-1] in chancelleryBlock if chancelleryBlock[-1] in model.key_to_index]

# Create a list of the words corresponding to these vectors
words_filtered = [word for word in mainDataWordsList if word in model.key_to_index]

# Zip the words together with their vector representations
word_vec_zip = zip(words_filtered, vector_list)

# Cast to a dict, so we can turn it into a DataFrame
word_vec_dict = dict(word_vec_zip)
df = pd.DataFrame.from_dict(word_vec_dict, orient='index')
print(print("Printing head of data frame:\n", df.head(10)))


def document_vector(word2vec_model, doc):
    vectors = []
    # Für alle Wörter im aktuell betrachteten Dokument...
    for word in doc:
        # ...wenn das Wort im NLM vorkommt...
        if word in model:
            value = model[word]
            # ...speichern des Vektors, den das Modell für das aktuell betrachtete gespeichert hat, in einer Liste
            if value.size > 0:
                vectors.append(model[word])
            else:
                print("Folgendes Wort", word, "hat im Model den Vektor,", model[word], "- ist also leer.")
    # Wenn Liste der Vektoren nicht leer ist, Rückgabe des Mittelwerts als Dokumentvektor
    if vectors:
        return np.mean(vectors, axis=0)


def annotiertes_vorgehen():
    featureClusterEmpathie = []
    featureClusterWortwahlKomplexitaet = []

    # Filtering the relevant features from the feature collection of all chancellery websites
    # & appending a list of "currentChancelleryName, currentHTMLText,currentFeatureExpression" to the relevant featureClusterList
    chancelleryNameChecklist1 = []
    for i in range(len(chancelleryHTMLtexts)):
        currentChancelleryBlock = chancelleryHTMLtexts[i]
        currentChancelleryName = currentChancelleryBlock[0]

        currentHTMLText = currentChancelleryBlock[1]
        currentFeaturesList = currentChancelleryBlock[2]
        if [] in currentFeaturesList:
            continue
        chancelleryNameChecklist1.append([currentChancelleryName, currentFeaturesList])
        for k in range(len(currentFeaturesList)):
            currentFeatureGroup = currentFeaturesList[k]
            currentFeatureId = currentFeatureGroup[0]
            currentFeatureExpression = currentFeatureGroup[1]
            if currentFeatureId == "F8":  # F8 = Empathie
                featureClusterEmpathie.append([currentChancelleryName, currentHTMLText, currentFeatureExpression])
            if currentFeatureId == "F23":  # F23 = WortwahlKomplexitaet
                featureClusterWortwahlKomplexitaet.append([currentChancelleryName, currentHTMLText, currentFeatureExpression])
    print("Recognized", len(featureClusterEmpathie), "HTML texts with a feature expression of: Empathie")
    print("Recognized", len(featureClusterWortwahlKomplexitaet), "HTML texts with a feature expression of: Wortwahl Komplexitaet")
    print("ChancelleryNameChecklist1:", chancelleryNameChecklist1)

    # Calculating the word vectors for each word in each document in both feature lists
    documentVectorsEmpathie = []
    for featureGroup in featureClusterEmpathie:
        currentChancelleryName = featureGroup[0]
        currentHTMLText = featureGroup[1]
        currentFeatureExpression = featureGroup[2]
        wordVectors = []
        for word in currentHTMLText:
            if word in model:
                wordVectors.append(model[word])
        documentVectorsEmpathie.append([currentChancelleryName, wordVectors, currentFeatureExpression])
    print("Calculated a number of", len(documentVectorsEmpathie), "document vectors of feature Empathie")

    documentVectorsWortwahlKomplexitaet = []
    for featureGroup in featureClusterWortwahlKomplexitaet:
        currentChancelleryName = featureGroup[0]
        if currentChancelleryName != ' ':
            currentHTMLText = featureGroup[1]
            currentFeatureExpression = featureGroup[2]
            wordVectors = []
            for word in currentHTMLText:
                if word in model:
                    wordVectors.append(model[word])
            documentVectorsWortwahlKomplexitaet.append([currentChancelleryName, wordVectors, currentFeatureExpression])
    print("Calculated a number of", len(documentVectorsWortwahlKomplexitaet), "document vectors of feature WortwahlKomplexitaet")
    for eintrag in documentVectorsEmpathie:
        print("Kanzlei", eintrag[0], "hat", len(eintrag[1]), "Wortvektoren.")

    # TODO: Centroid-Vektor jedes einzelnen Dokuments

    # print([group[0] for group in documentVectorsEmpathie])

    def calculate_centroid(word_vectors):
        # Aufsummieren aller Vektoren
        # print("Trying to sum up the following number of word vectors", len(word_vectors))
        # total_vector = np.sum(word_vectors, axis=0)
        #
        # if np.any(np.isnan(total_vector)) or np.any(np.isnan(word_vectors)):
        #     centroid = np.zeros(total_vector.shape)
        #
        # else:
        #     try:
        #         centroid = total_vector / len(word_vectors)
        #     except FloatingPointError:
        #         centroid = np.zeros(total_vector.shape)
        #    # Berechnen Sie den Durchschnitt der Vektoren
        #    # centroid = total_vector / len(word_vectors)
        print("Len of word_vectors", len(word_vectors))

        word_vectors_array = np.array(word_vectors)

        # Berechnen des Median jedes Elements im Array
        centroid = np.median(word_vectors_array, axis=0)

        return centroid

    # Calculating the centroid vector for each document in the document vector lists for both features
    print("Computing centroidVectors for feature Empathie")
    centroid_vectorsEmpathie = []
    for wordVectorGroup in documentVectorsEmpathie:
        currentChancelleryName = wordVectorGroup[0]
        if currentChancelleryName != ' ':
            currentWordVectors = wordVectorGroup[1]
            currentFeatureExpression = wordVectorGroup[2]
            # if currentChancelleryName == "christianKoch":
            #    print("\n\n\n!!! currentWordVectors of ChristianKoch:", currentWordVectors)
            # print("sum(currentWordVectors):", sum(currentWordVectors), "/ len(currentWordVectors):", len(currentWordVectors))
            currentDocumentsCentroidVector = calculate_centroid(currentWordVectors)  # sum(currentWordVectors) / len(currentWordVectors)
            centroid_vectorsEmpathie.append([currentChancelleryName, currentDocumentsCentroidVector, currentFeatureExpression])

    print("Computing centroidVectors for feature WortwahlKomplexitaet")
    centroid_vectorsWortwahlKomplexitaet = []
    for wordVectorGroup in documentVectorsWortwahlKomplexitaet:
        currentChancelleryName = wordVectorGroup[0]
        currentWordVectors = wordVectorGroup[1]
        currentFeatureExpression = wordVectorGroup[2]
        if len(currentWordVectors) > 0:
            currentDocumentsCentroidVector = calculate_centroid(currentWordVectors)  # sum(currentWordVectors) / len(currentWordVectors)
            # if currentChancelleryName == "gansel":
            #    print("Gansel__ sum(currentWordVectors):", sum(currentWordVectors), "/", "len(currentwordvectors):", len(currentWordVectors))
            centroid_vectorsWortwahlKomplexitaet.append([currentChancelleryName, currentDocumentsCentroidVector, currentFeatureExpression])

    prominentChancelleriesForFeatures = [["F8", "christianKoch", 1], ["F23", "gansel", 1]]  # Alternativ für komplexitaetWortwahl: schmittHaensler

    # For testing purposes uniting the centroid vectors
    unitedCentroidVectors = []
    for chancelleryName, centroidVector, featureExpression in centroid_vectorsEmpathie:
        for n, v, f in centroid_vectorsWortwahlKomplexitaet:
            if n == chancelleryName:
                unitedCentroidVectors.append([chancelleryName, centroidVector + v, [featureExpression, f]])

    # print("UnitedVectorsList:", unitedCentroidVectors)

    # Calculating the distances between all documents of the same feature
    # For testing purposes: Intially no calculation of the distances from each document's centroid vector to the prominent chancelleries' ones
    # But calculation the distances for all centroid vectors of each feature category TODO: Could alter this here for testing purposes

    # distancesEmpathie = euclidean_distances(list([vectorGroup[1] for vectorGroup in centroid_vectorsEmpathie]))
    # distancesWortwahlKomplexitaet = euclidean_distances(list(vectorGroup[1] for vectorGroup in centroid_vectorsWortwahlKomplexitaet))
    def calculate_single_feature_distances():
        distancesEmpathie = []
        for vectorGroup in centroid_vectorsEmpathie:
            currentChancelleryName = vectorGroup[0]
            currentCentroidsDistances = euclidean_distances(vectorGroup[1])
            currentFeatureExpression = vectorGroup[2]
            distancesEmpathie.append([currentChancelleryName, currentCentroidsDistances, currentFeatureExpression])

        distancesWortwahlKomplexitaet = []
        for vectorGroup in centroid_vectorsWortwahlKomplexitaet:
            currentChancelleryName = vectorGroup[0]
            currentCentroidsDistances = euclidean_distances(vectorGroup[1])
            currentFeatureExpression = vectorGroup[2]
            distancesWortwahlKomplexitaet.append([currentChancelleryName, currentCentroidsDistances, currentFeatureExpression])

    # Extracting names, vectors and feature expressions from the list unitedCentroidVectors
    unifiedVectorNames = [vectorGroup[0] for vectorGroup in unitedCentroidVectors]
    unifiedCentroidVectorValues = [vectorGroup[1] for vectorGroup in unitedCentroidVectors]
    unifiedCentroidVectorFeatureExpressions = [vectorGroup[2] for vectorGroup in unitedCentroidVectors]

    # Calculating vector distances
    unitedCentroidVectorsDistances = euclidean_distances(unifiedCentroidVectorValues)

    # Setze alle Distanzen auf einen minimalen positiven Wert
    unitedCentroidVectorsDistances[unitedCentroidVectorsDistances <= 0] = np.finfo(float).eps

    # Converting the distances to logarithmic distances
    logDistances = np.log(unitedCentroidVectorsDistances)

    vmin = np.min(logDistances)
    vmax = np.max(logDistances)

    # Calculating list of indices of all vector names
    nameIndices = list(range(len(unifiedVectorNames)))

    # Creating a DataFrame with names as indices and distances as values
    df = pd.DataFrame(logDistances, index=unifiedVectorNames, columns=unifiedVectorNames)

    plt.rcParams.update(
        {
            'text.usetex': False,
            'font.family': 'stixgeneral',
            'mathtext.fontset': 'stix',
        }
    )

    plt.figure(figsize=(8, 8))

    # Plotting the distance matrix with MatPlotLib
    plt.imshow(df, cmap='Blues', vmin=vmin, vmax=vmax)

    # Adding the distances' names as nametags
    plt.xticks(nameIndices, unifiedVectorNames, rotation=45)
    plt.yticks(nameIndices, unifiedVectorNames)

    plt.show()

    def formerClusteringApproach():
        # Migrating both centroid vector lists
        distancesMigrated = []
        for vectorGroup in distancesEmpathie:
            for secondVectorGroup in distancesWortwahlKomplexitaet:
                currentChancelleryName = vectorGroup[0]
                firstCentroidVector = vectorGroup[1]
                firstFeatureExpression = vectorGroup[2]
                secondCentroidVector = secondVectorGroup[1]
                secondFeatureExpression = secondVectorGroup[2]
                if vectorGroup[0] == secondVectorGroup[0]:
                    distancesMigrated.append([currentChancelleryName, [firstCentroidVector, secondCentroidVector], [firstFeatureExpression, secondFeatureExpression]])
        print("DistancesMigrated:", distancesMigrated)
        # Extracting the distances out of the migrated list
        X = [vectorGroup[1] for vectorGroup in distancesEmpathie]  # in distancesMigrated
        Y = [vectorGroup[1] for vectorGroup in distancesWortwahlKomplexitaet]
        XY = []
        for firstDistance in X:
            for secondDistance in Y:
                XY.append([firstDistance, secondDistance])
        # Setting the threshold
        threshold = 0.5

        # Defining the number of clusters
        k = 2

        # Creating the KMeans model
        km = KMeans(n_clusters=k)

        # Fitting the model to the data
        km.fit(XY)

        # Predicting the clusters for each document
        clusters = km.predict(XY)
        print("anzahl cluster:", len(clusters))

        # Printing the clusters
        # TODO: Fehler beheben: Hier wird beim Ausgeben der Cluster auf die Anzahl in distancesMigrated gesetzt. Die Cluster werden aber auf X & Y zusammen berechnet. Auflösen!
        for i, cluster in enumerate(clusters):
            print(f"{distancesMigrated[i][0]}: {cluster}")

        # Creating a scatterplot to visualize the clusters
        plt.scatter(X, Y, c=clusters, cmap='viridis')  # plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis')
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title("Document Clusters")
        plt.show()


# unannotiertes_vorgehen()


annotiertes_vorgehen()


def unannotiertes_vorgehen():
    # documentVectors = document_vector(model, [chancelleryBlock[-1] for chancelleryBlock in mainDataWordsList])
    # documentVectors = document_vector(model, [chancelleryHTML for chancelleryHTML in chancelleriesWordsList])
    print("Calculating document vectors...")
    documentVectors = []
    print("ChancelleryGroup 1, chancelleryHTMLtext 1", chancelleryHTMLtexts[1][1][:200])
    for chancelleryHTMLgroup in chancelleryHTMLtexts:
        chancelleryHTML = chancelleryHTMLgroup[1]
        documentVectors.append([chancelleryHTMLgroup[0], document_vector(model, chancelleryHTML)])
    print("Printing a slice of 2 of all", len(documentVectors), " document vectors:\n", documentVectors[:2])
    # TODO: Empty slice/ division durch 0 vor printing der slices der Dokumentvektoren beheben
    # print("Example of calculating centroid vector: documentVector")
    print(
        "\nCalculating centroid vectors/ summing up all vectors of each document")  # by", sum([documentVector for documentVector in documentVectors[1]]), "/", len(documentVectors))

    # documentVectorSums = []
    # for documentVectorGroup in documentVectors:
    #     currentChancelleryName = documentVectorGroup[0]
    #     currentChancelleryVectors = documentVectorGroup[1]  # type: array
    #     currentChancelleryVectorsSum = np.sum(currentChancelleryVectors)
    #     documentVectorSums.append([currentChancelleryName, currentChancelleryVectorsSum])
    # print(len(documentVectorSums), "document vector sums created. These are the first 5", documentVectorSums[:4])

    centroidVectors = []

    for i in range(len(documentVectors)):
        chancelleryHTMLgroup = documentVectors[i]
        documentVectorsOfCurrentHTML = chancelleryHTMLgroup[1]
        if documentVectorsOfCurrentHTML is not None:
            if documentVectorsOfCurrentHTML.shape:
                centroidVector = np.sum(documentVectorsOfCurrentHTML) / documentVectorsOfCurrentHTML.size  # .shape[0]
                centroidVectors.append([chancelleryHTMLgroup[0], centroidVector])
    print("Centroid vectors list has a total of", len(centroidVectors), "entries. Here come the first 3", centroidVectors[:3])

    # Extrahieren der klassen-prototypischen Centroid-Vektoren:
    # knoff: Durchsetzungswillen
    # christianKoch: Unerfahrene Mandanten abholend
    centroidVectorKnoff = [specificCentroidVector for specificCentroidVector in centroidVectors if specificCentroidVector[0] == "knoff"][0]
    centroidVectorChristianKoch = [specificCentroidVector for specificCentroidVector in centroidVectors if specificCentroidVector[0] == "christianKoch"][
        0]

    print("Kanzlei Knoff ist prototypisch für einen kämpferischen Stil. Knoff hat folgenden centroid-Vektor", centroidVectorKnoff[1], "vom Typ",
          type(centroidVectorKnoff[1]))
    distanceCentroidVectorsToKnoff = []
    distanceCentroidVectorsToChristianKoch = []
    for vectorGroup in centroidVectors:
        currentChancelleryName = vectorGroup[0]
        currentCentroidVector = vectorGroup[1]
        print("Schaue mir Kanzlei", currentChancelleryName, "an. Die Kanzlei-Seite hat den Centroid-Vektor", currentCentroidVector, "vom Typ",
              type(currentCentroidVector))
        if currentCentroidVector.size != centroidVectorKnoff[1].size:
            print("Resizing vectors to be the same length")
            currentCentroidVector = np.resize(currentCentroidVector, len(centroidVectorKnoff[1]))
        # currentSizeDifference = len(centroidVectorKnoff[1]) - len(currentCentroidVector)
        # currentCentroidVector.extend()

        currentCentroidVectorDistanceToKnoff = np.linalg.norm(
            currentCentroidVector - centroidVectorKnoff[1])  # distance.cosine(currentCentroidVector, centroidVectorKnoff[1])
        currentCentroidVectorDistanceToChristianKoch = np.linalg.norm(currentCentroidVector - centroidVectorChristianKoch[1])
        distanceCentroidVectorsToKnoff.append([currentChancelleryName, currentCentroidVectorDistanceToKnoff])
        distanceCentroidVectorsToChristianKoch.append([currentChancelleryName, currentCentroidVectorDistanceToChristianKoch])
        # print("Vector currentCentroidVector:", currentCentroidVector, "and centroidVectorKnoff[1]:", centroidVectorKnoff[1])

    print("Distanzen der Kanzleien zu Knoff im Vektorraum:", distanceCentroidVectorsToKnoff)

    distanceCentroidVectorsToKnoff_sorted = sorted(distanceCentroidVectorsToKnoff, key=lambda x: x[1])
    distanceCentroidVectorsToChristianKoch_sorted = sorted(distanceCentroidVectorsToChristianKoch, key=lambda x: x[1])
    print("Sortierte Distanzen der Kanzleien zu Knoff:", distanceCentroidVectorsToKnoff_sorted)
    print("Sortierte Distanzen der Kanzleien zu ChristianKoch:", distanceCentroidVectorsToChristianKoch_sorted)

    winsound.PlaySound('SystemAsterisk.wav', winsound.SND_FILENAME)

    def cluster_and_plot(vector_blocks, vector_distance_blocks):
        vectors = []
        vector_distances = []
        # vector_blocks = [['gansel', 7.270783185958857e-05], ['illner', 0.0006425455212593074], ['heinz', 0.0004028531908988951]]
        vectorTupleList = []
        vectorDistanceList = []

        for i in range(len(vector_blocks)):
            vectorBlock = vector_blocks[i]
            vectorDistanceBlock = vector_distance_blocks[i]
            currentVector = np.array(vectorBlock[1])
            currentDistance = np.array(vectorDistanceBlock[1])
            vectorTupleList.append((currentVector, np.resize(0.00000000, currentVector.shape)))
            vectorDistanceList.append(currentDistance)

        # tupleList = [(x[1], np.resize(0.00000000, x[1].shape)) for x in vector_blocks]
        print("TupleList:", vectorTupleList)
        vectorArrays = [np.array(y) for y in vectorTupleList]
        print("VectorArrays:", vectorArrays)

        # for k in range(len(vector_blocks)):
        #     vectorBlock = vector_blocks[k]
        #     print("Vector block", k, "consists of:", vectorBlock, "vom Typ", type(vectorBlock[1]))
        #     if vectorBlock[1] == [] or vectorBlock[1].size == 0:
        #         print("Found empty vectorBlock. Skipping:", vectorBlock)
        #     elif vectorBlock[1].ndim == 0:
        #         vectorArray = np.array([[vectorBlock[1], 0.00]])
        #         print("Working on the following vectorArray:", vectorArray)
        #         x_value = vectorArray[:, 0]
        #         y_value = vectorArray[:, 1]
        #         y_value = np.resize(y_value, x_value.shape)
        #         vectors.append(np.array([[x_value, y_value]]))
        #         print("Got vector block", vectorBlock, " - creating the following array:", vectorArray)
        #     elif vectorBlock[1].ndim == 1:
        #         vectors.append(vectorBlock[1].reshape(1, -1))
        #     elif vectorBlock[1].ndim > 1:
        #         vectors.append(vectorBlock[1])
        for vectorDistanceBlock in vector_distance_blocks:
            vector_distances.append(vectorDistanceBlock[1])

        # print("Passing the following vectors to KMeans:", vectors)

        kmeans = KMeans(n_clusters=3)
        kmeans.fit(vectorArrays)

        vectorDistancesResized = []
        for distanceElement in vectorDistanceList:
            vectorDistancesResized.append(np.resize(distanceElement, vectorArrays[0].shape))
        print("Size of 'vectorArrays:", len(vectorArrays), "| size of 'vectorDistancesResized':", len(vectorDistancesResized))  # , "| size of 'labels':", len(labels_))
        print("VectorArrays:", vectorArrays, "\nVectorDistancesResized:", vectorDistancesResized)
        plt.scatter(vectorArrays, vectorDistancesResized, c='b', marker='+', cmap="rainbow")  # c=kmeans.labels_, cmap="rainbow")

        # for v, name in enumerate(vector_blocks):
        #     plt.annotate(name, (vector_blocks[v][0], vector_blocks[v][1]))

        plt.show()

    cluster_and_plot(centroidVectors, distanceCentroidVectorsToKnoff)


def plotting():
    # Initialize t-SNE
    tsne = TSNE(n_components=2, init='random', random_state=10, perplexity=100)

    # Use only 400 rows to shorten processing time
    tsne_df = tsne.fit_transform(df[:400])

    sns.set()
    # Initialize figure
    fig, ax = plt.subplots(figsize=(11.7, 8.27))
    sns.scatterplot(tsne_df[:, 1], alpha=0.5)  # entfernt wurde tsne_df[:, 0],

    # Import adjustText, initialize list of texts
    from adjustText import adjust_text

    texts = []
    words_to_plot = list(np.arange(0, 400, 5))  # vorher , , 10

    # Append words to list
    for word in words_to_plot:
        texts.append(plt.text(tsne_df[word, 0], tsne_df[word, 1], df.index[word], fontsize=14))

    # Plot text using adjust_text (because overlapping text is hard to read)
    adjust_text(texts, force_points=0.4, force_text=0.4,
                expand_points=(2, 1), expand_text=(1, 2),
                arrowprops=dict(arrowstyle="-", color='black', lw=0.5))

    plt.show()


# Our earlier preprocessing was done when we were dealing only with word vectors
# Here, we need each document to remain a document
def preprocess(text):
    text = text.lower()
    doc = word_tokenize(text)
    doc = [word for word in doc if word not in stopWords]
    doc = [word for word in doc if word.isalpha()]
    return doc


# Function that will help us drop documents that have no word vectors in word2vec
def has_vector_representation(word2vec_model, doc):
    """check if at least one word of the document is in the
    word2vec dictionary"""
    return not all(word not in word2vec_model.vocab for word in doc)


# Filter out documents
def filter_docs(corpus, texts, condition_on_doc):
    """
    Filter corpus and texts given the function condition_on_doc which takes a doc. The document doc is kept if condition_on_doc(doc) is true.
    """
    number_of_docs = len(corpus)

    if texts is not None:
        texts = [text for (text, doc) in zip(texts, corpus)
                 if condition_on_doc(doc)]

    corpus = [doc for doc in corpus if condition_on_doc(doc)]

    print("{} docs removed".format(number_of_docs - len(corpus)))

    return corpus, texts
