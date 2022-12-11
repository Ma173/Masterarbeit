import gensim
from gensim import utils
from gensim.test.utils import datapath
from gensim.models import KeyedVectors, word2vec
# from gensim.models.wrappers import FastText
import time
from datetime import datetime
import winsound
from matplotlib import pyplot as plt
from nltk import word_tokenize
from sklearn.manifold import TSNE
from joblib import Parallel, delayed
from bs4 import BeautifulSoup

import pandas as pd

import seaborn as sns
# import matplotlib.pyplot as plt
import numpy as np

from nltk.corpus import stopwords

stopWords = set(stopwords.words('german'))
# import nltk
# import xgboost as xgb
# import unicodedata

mainDataWordsList = []
chancelleriesWordsList = []
chancelleryHTMLtexts = []


def preprocessing(corpus_path):
    valuableFeatureIds = ["F8", "F22", "F23"]

    chancelleryBlocksRaw = open(datapath(corpus_path), encoding='unicode_escape').read().split("__________")  # vorher: encoding="utf-8"
    chancelleryBlocks = []

    # Erstellung einer Liste mit allen Kanzlei-Blöcken
    # und je BLock einer Liste aller Zeilen, je Zeile eine Liste aller Einheiten (Feature-Nummer, Fundort, etc.)

    global currentChancelleryName, currentChancelleryHTML, currentLine, currentLineSplitted, newChancelleryBlock

    # Pro Kanzlei-Block
    for i in range(len(chancelleryBlocksRaw)):
        currentChancelleryBlockSplitted = []
        currentChancelleryBlock = chancelleryBlocksRaw[i].splitlines()
        newChancelleryBlock = []
        currentChancelleryWordsList = []

        # Pro Zeile in Kanzlei-Block
        for k in range(len(currentChancelleryBlock)):
            currentLine = currentChancelleryBlock[k]
            currentFeatureId = currentLine.split(" ")[0]
            currentLineSplitted = []

            if currentFeatureId in valuableFeatureIds:  # currentLine.startswith("F"):
                featureInfo = currentLine[:currentLine.find(" |")].split(" ")
                featureInfoActualData = currentLine[currentLine.find(" |"):]
                tokens = utils.simple_preprocess(featureInfoActualData)
                words = [word.lower() for word in tokens if word.isalpha()]

                words = [word for word in words if not word in stopWords]
                for word in words:
                    mainDataWordsList.append(word)
                    currentChancelleryWordsList.append(word)

                for featureInfoElement in featureInfo:
                    currentLineSplitted.append(featureInfoElement)

                currentLineSplitted.append(words)  # (featureInfoActualData)
                newChancelleryBlock.append(currentLineSplitted)
            elif currentLine.startswith("<"):
                currentChancelleryHTML = currentLine
                currentChancelleryHTMLclean = BeautifulSoup(currentChancelleryHTML, "html.parser").get_text()


            elif len(currentLine.split(" ")) < 2:
                currentChancelleryName = currentLine
                if currentChancelleryName:
                    newChancelleryBlock.append(currentChancelleryName)

        chancelleryBlocks.append([newChancelleryBlock, currentChancelleryHTMLclean])  # ,currentChancelleryHTML])
        chancelleriesWordsList.append([currentChancelleryName, currentChancelleryWordsList])
        chancelleryHTMLtexts.append([currentChancelleryName, currentChancelleryHTMLclean])
    return chancelleryBlocks


# TODO: Dran denken, dass die Elemente 5 & 6 in den Kanzleizeilen optional sind, weil 3 & 4 = 5 & 6 sein können (Textquelle = Fundstelle)

startPreprocessing = time.time()
print("Starting to preprocess the annotated data...")
mainData = preprocessing(r'C:/Users/Malte/Dropbox/Studium/Linguistik Master HHU/Masterarbeit/websitesTextsReformatted.txt')

print("Finished preprocessed data. Time elapsed:", round(((time.time() - startPreprocessing) / 60), 2))


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
    model = gensim.models.KeyedVectors.load(r'B:/Python-Projekte/Masterarbeit/' + possibleModelsToLoad[modelToLoad], mmap='r')

else:
    with Parallel(n_jobs=-1) as parallel:
        model = gensim.models.KeyedVectors.load_word2vec_format(r'B:/Python-Projekte/Masterarbeit/' + possibleModelsToLoad[modelToLoad], binary=False,
                                                                encoding='unicode escape', workers=4)  # ,  workers=parallel)

model.fill_norms()
# model.save(modelToLoad + "_loaded")

timeSinceLoadingModel = round((time.time() - startLoadingModel) / 60, 3)
print("Finished loading model. Time elapsed:", timeSinceLoadingModel, "minutes.")
winsound.PlaySound('SystemAsterisk.wav', winsound.SND_FILENAME)
startComputingModelInfo = time.time()
print("Starting to compute some model info...")
model.fill_norms()
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


def document_vector(word2vec_model, doc):
    # remove out-of-vocabulary words
    # doc = [word for word in doc if word in model.key_to_index]
    vectors = []
    for word in doc:
        if word in model:
            vectors.append(model[word])
    return np.mean(vectors, axis=0)


# documentVectors = document_vector(model, [chancelleryBlock[-1] for chancelleryBlock in mainDataWordsList])
# documentVectors = document_vector(model, [chancelleryHTML for chancelleryHTML in chancelleriesWordsList])
print("Calculating document vectors...")
documentVectors = []
for chancelleryHTML in chancelleryHTMLtexts:
    documentVectors.append(document_vector(model, chancelleryHTML))
print("Printing a slice of 5 of all", len(documentVectors), " document vectors:\n", documentVectors[:4])

centroidVector = sum(documentVectors) / len(documentVectors)
print("Centroid vector:", centroidVector)


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

    return (corpus, texts)

# titles_list = [title[0] for title in mainData]
# # Preprocess the corpus
# # TODO: Klären warum da die Titel im Tutorial preprocessed wurden. Ich will doch die HTML-Texte preprocessen?!
# # TODO: Rausfinden wie ich jetzt die Umgebungsvariablen von venv richtig anwende, damit ich dieses package zum schnelleren model laden installieren kann
# corpus = [preprocess(chancelleryBlock[1][-1]) for chancelleryBlock in mainData]
#
# # Remove docs that don't include any words in W2V's vocab
# corpus, titles_list = filter_docs(corpus, titles_list, lambda doc: has_vector_representation(model, doc))
#
# # Filter out any empty docs
# corpus, titles_list = filter_docs(corpus, titles_list, lambda doc: (len(doc) != 0))
# x = []
# for doc in corpus:  # append the vector for each document
#     x.append(document_vector(model, doc))
#
# X = np.array(x)  # list to array
#
# plotting()
