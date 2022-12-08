from typing import List

import self as self
from gensim.test.utils import datapath
from gensim import utils
import gensim.models
import tempfile

# If the following variable is set to False, the model is created as a new one.
# If the variable is set to True, the model is loaded from disk
loadModel = False
retrieveVocabulary = False
model = ""
chancelleryNames = []
chancelleryHTMLs = []
chancelleryFeatures = []


class ChancelleryCorpusOld:
    # An iterator that yíelds chancellery blocks (lines of strings)
    def __iter__(self):
        # corpus_path = datapath(r'C:\Users\Malte\Dropbox\Studium\Linguistik Master HHU\Masterarbeit\websitesTexts (2)')
        corpus_path = datapath(r'C:/Users/Malte/Dropbox/Studium/Linguistik Master HHU/Masterarbeit/websitesTexts (2).txt')
        chancelleryBlocks = open(corpus_path, encoding="utf-8").read().split("__________")
        print("Found", len(chancelleryBlocks), "chancellery blocks in the corpus.")
        for i in range(len(chancelleryBlocks)):
            print("Checking block", i + 1)
            chancelleryBlock = chancelleryBlocks[i]
            if len(chancelleryBlock.splitlines()) > 1:
                # print("Length of chancellery block:",len(chancelleryBlock.splitlines()))
                chancelleryName = chancelleryBlock.splitlines()[0]
                chancelleryHTML = chancelleryBlock.splitlines()[1]
                chancelleryFeature = chancelleryBlock.splitlines()[2:]
                # print("Preprocessing line:", chancelleryHTML)
                # yield utils.simple_preprocess(chancelleryHTML)
                yield from (chancelleryName, utils.simple_preprocess(chancelleryHTML), chancelleryFeature)
        # for line in open(corpus_path):
        #    yield utils.simple_preprocess(line)


class ChancelleryCorpus2:
    def preprocessing(self):
        chancelleryBlockProcessed = []
        corpus_path = datapath(r'C:/Users/Malte/Dropbox/Studium/Linguistik Master HHU/Masterarbeit/websitesTexts (2).txt')
        chancelleryBlocks: List[str] = open(corpus_path, encoding="utf-8").read().split("__________")
        print("Found", len(chancelleryBlocks), "chancellery blocks in the corpus.")
        for i in range(len(chancelleryBlocks)):
            print("Checking block", i + 1)
            chancelleryBlock = chancelleryBlocks[i]
            if len(chancelleryBlock.splitlines()) > 1:
                # print("Length of chancellery block:",len(chancelleryBlock.splitlines()))
                chancelleryName = chancelleryBlock.splitlines()[0]
                chancelleryHTML = chancelleryBlock.splitlines()[1]
                chancelleryFeatureLines = chancelleryBlock.splitlines()[2:]
                # print("Preprocessing line:", chancelleryHTML)
                # yield utils.simple_preprocess(chancelleryHTML)
                chancelleryNames.append(chancelleryName)
                chancelleryHTMLs.append(chancelleryHTML)
                chancelleryFeatures.append(chancelleryName)
                chancelleryBlockProcessed += [chancelleryName, utils.simple_preprocess(chancelleryHTML), chancelleryFeatureLines]
                # TODO: Klären was preprocess genau macht und dann entscheiden, ob ich es wirklich auf den HTML-Text oder eher auf die Features anwenden will.
                #  Danach schauen wie ich das an die sentences des Modells übergebe (oder muss es überhaupt an sentences sein? -> das Tutorial dazu von Word2Vec checken) dann an Rafael schreiben (Danke + wie machen wir weitewr, auch textuell?)
        return chancelleryBlockProcessed


if loadModel is False:
    print("Step 1a started: Calculating new model based on preprocessed sentences")
    # sentences = GensimCorpus()
    # a, b, c = ChancelleryCorpus.__iter__(self)
    # print(a)
    # print(b)
    # print(c)
    # print([v for v in ChancelleryCorpus.__iter__(self)])
    # sentences = [v for v in ChancelleryCorpus.__iter__(self)]
    chancelleryBlocks = ChancelleryCorpus2.preprocessing(self)

    # for i in range(len(chancelleryBlocks)):
    #     block = chancelleryBlocks[i]

    # print(sentences)
    sentences2 = chancelleryHTMLs

    model = gensim.models.Word2Vec(sentences=sentences2, min_count=10, workers=4)
    print("Step 1a finished. New model based on preprocessed sentences calculated.")

    print("Step 1b started: Saving newly calculated model.")
    with tempfile.NamedTemporaryFile(prefix='gensim-model-', delete=False) as tmp:
        temporary_filepath = tmp.name
        final_filepath = 'C:/Users/Malte/Dropbox/Studium/Linguistik Master HHU/Masterarbeit/gensim-model-01'
        model.save(final_filepath)
    print("Step 1b finished: Newly calculated model saved.")


elif loadModel is True:

    print("Step 1: Loading previously created model")
    model = gensim.models.Word2Vec.load(r'B:\Python-Projekte\Masterarbeit/gensim-model-u5oewsr5')
    print("Step 1 finished. Previously created model loaded.")
    # print("Step 2: Importing wordvector bib from run_word2vec")
    # from run_word2vec import wv

print("Step 2 started: Calculating the model's word vectors.")
wv = model.wv
print("Status of word vectors:", wv)
print("Step 2 finished: Model's word vectors calculated.")

# vec_king = model.wv['Kanzlei']

print(wv.vocab)

if retrieveVocabulary is True:
    for index, word in enumerate(wv.index_to_key):
        if index == 10:
            break
        print(f"word #{index}/{len(wv.index_to_key)} is {word}")
