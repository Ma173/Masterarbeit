import requests, re
from tryout import nGram
from toolbox import sortDict, first_with_x_count, loadFromFile, similarityOfStrings, saveListToFile, loadListOfTuplesFromFile, getShortestItem, frequency
from chancelleryURLs import websitesDictWithChancelleryName
from userInteraction import userSuggestedFeatures, userinput

from gensim.test.utils import datapath
import gensim.py

from gensim.test.utils import datapath
from gensim import utils
import gensim.models

class MyCorpus:
  def _iter_(self):
    corpus_path = datapath ('lee_background.cor')
    for line in open(corpus_path):
      yield utils.simple_preprocess(line)

sentences = MyCorpus()
model = gensim.models.Word2Vec(sentences=sentences)

vec_king = model.wv['king']

for index, word in enumerate(wv.index_to_key):
  if index == 10:
    break
  print(f"word #{index}/{len(wv.index_to_key)} is {word}")

# The function to get the text of a single website
def getSingleWebsiteData(url):
    print("\tGathering single website data (text)...")
    #print(url)
    url = url if url.startswith('http') else ('http://' + url)
    r= requests.get(url)
    r.encoding = r.apparent_encoding
    websiteText = r.text
    websiteTextClean = ""
    for line in websiteText.splitlines():
        if not line.endswith("</html>"):
            websiteTextClean += line
        elif line.endswith("</html>"):
            break
    print("\tDone gathering data.")
    return websiteTextClean


def getMultipleWebsiteData(urlCollection):
    urllist = []
    typeOfUrlCollection = ""
    #print(type(urlCollection))
    print("Assessing type of URL collection named '{}'".format(urlCollection))
    if (urlCollection is list) or (isinstance(urlCollection, list)):
        print("Collection of urls as of list type detected")
        urllist = urlCollection[:]
        typeOfUrlCollection = list
    elif isinstance(urlCollection, dict) or urlCollection is dict:
        print("Collection of urls as of dictionary type detected")
        for key, value in urlCollection.items():
            pair = [key, value]
            urllist.append(pair)
        typeOfUrlCollection = dict
    else:
        print(
            "Type of collection not recognized. Type was {} No website retrieved from web."
            .format(type(urlCollection)))
        urllist = urlCollection[:10]
        print("First items of collection were: {}".format(urllist[:5]))
        typeOfUrlCollection = None
    print("Gathering text data of {} websites...".format(len(urllist)))
    websitesTexts = []
    # Iterating through all websites (Website Name & actual url) and saving a list of website name and website text to a list
    for i in range(len(urllist)):
        website = urllist[i]
        #print("website=urllist[i] is: {}".format(urllist[i]))
        if i % 1 == 0:
            print("\n\t{} of {}: {}".format(i+1, len(urllist), website))
        if (typeOfUrlCollection is list and len(website) >= 1):
            try:
                websiteText = getSingleWebsiteData(website)
                if len(websiteText) > 1:
                    websitesTexts.append((website, websiteText))
                    #print("Appending the following: {}".format((website,websiteText[:50])))
            except:
                print("Website not reachable. Moving on to the next.")

        elif "http" in website[0]:
            #print("TEST2")

            websitesTexts.append((website[1],
                                  getSingleWebsiteData(website[0])))
            #print("Appending the following: {}".format((website[1],getSingleWebsiteData(website[0])[:50])))
        elif "http" in website[1]:
            #print("TEST3")
            websitesTexts.append((website[0],
                                  getSingleWebsiteData(website[1])))
            #print("Appending the following: {}".format((website[0],getSingleWebsiteData(website[1])[:50])))
        if len(websitesTexts) % 5 == 0:
            saveListToFile(websitesTexts, "websitesTexts.txt")
    print("Done gathering text data of multiple websites.")
    return websitesTexts


# Getting the chancellery's name
def getChancelleryName(textfile):
    print("Getting chancellery name ...")
    chancelleryNameGroup = re.search(
        '<\/strong><\/h4>\n<h4><strong>(.*)<br \/><\/strong><\/h4', textfile)
    chancelleryName = chancelleryNameGroup.group(1)
    print("Done getting chancellery name.")
    return chancelleryName

overlappingFeaturesDict = {}

websitesFeaturesList = []

# Comparing a list of website texts
def textComparisonGetFeatures(texts):
    print("Extracting common features...")
    searchParametersArchive = [
        '=".*"', '="(.+)"', '<(.+)>', '="(.+)"', '(" alt=")(.+)(")'
    ]
    searchParameter3 = '(\w+=")([\w\d\s]+)(")'  #'(\w+=")(\s*\w+\s*)+(")'
    for text in texts:
        foundFeatures = re.findall(searchParameter3, text[1])
        #print("Foundall findet {} Treffer".format(len(foundFeatures)))
        # Converting the list of features into a set for removing duplicates easily and then converting it into a list again
        foundFeatures = list(set(foundFeatures))
        #print("Text [0] is {}; Text[1][:50] is {}".format(text[0],text[1][:50]))
        websiteName = text[0]
        websitesFeaturesList.append((websiteName, set(foundFeatures)))
        #print("\n- {} features in website text '{}' with a length of {}:\n{}\n\n".format(len(foundFeatures),websiteName[:100],len(text[1]),"---"))#foundFeatures))
        #for feature in foundFeatures:
        #  print("\t",feature)
    commonFeatures = []
    featuresListTexts = []
    # Gaining the common features of *all* texts through intersection. Might lead to very few hits
    for textSet in websitesFeaturesList:
        featuresListTexts.append(textSet[1])
        commonFeatures = set.intersection(
            *featuresListTexts
        )  #websitesFeaturesList[1],websitesFeaturesList[2])
    return commonFeatures


chancelleryUrls = loadFromFile("chancelleryURLs_2.txt").read().splitlines()

print("ChancelleryUrls length is: {}".format(len(chancelleryUrls)))

# Getting matching features of multiple websites texts by first gathering the websites' texts and then extracting common features

# SWITCH MODE TO UPDATE THE WEBSITES' TEXT FILES:
textImportMode = "RetrieveFromWeb" #"LoadFromFile" 
if textImportMode == "LoadFromFile":
  print("Loading website texts from file")
  websitesTexts = loadListOfTuplesFromFile("websitesTexts.txt")
elif textImportMode == "RetrieveFromWeb":
  print("Retrieving texts from web")
  websitesTexts = getMultipleWebsiteData(
        websitesDictWithChancelleryName)  #websitesDictWithChancelleryName)
  saveListToFile(websitesTexts,"websitesTexts.txt")

matchingFeatures = textComparisonGetFeatures(websitesTexts)
print("Features that match all websites: ", list(matchingFeatures)[:10])
websitesListOfFeaturesWithoutWebsitename = []
for i in range(len(websitesFeaturesList)):
    websitesListOfFeaturesWithoutWebsitename.append(
        list(websitesFeaturesList[i][1]))

featureFrequency = frequency(websitesListOfFeaturesWithoutWebsitename)
featureFrequencyTop = []
for featurePairs in featureFrequency:
    if featurePairs[1] >= 10:
        featureFrequencyTop.append(featurePairs)
# Printing out the three groups of the result of re.findall(searchParameter3, text[1]) followed by the calculated frequency
print("\nAll features that occur on at least 10 websites:")
for featureFreqPair in featureFrequencyTop:
    print(featureFreqPair)

print("End of list 'featureFreqPair', all features that occur on at least 10 websites")

# SET TRUE TO OFFER USER EVALUATION IN RUNNING CODE
usereval = False
if usereval == True:
  userinput(websitesTexts,websitesListOfFeaturesWithoutWebsitename)

import websitesAnnotated_2

def learningAlgorithmAnnotatedTexts():
  import os
  from os import walk
  from bs4 import BeautifulSoup
  #get working directory
  os.getcwd()
  os.listdir('/home/runner')
  print("\n§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§\nBeginning of learning algorithm\n")
  websiteFiles = []
  featureList = []
  folder = "websitesAnnotated_2"
  for (dirpath, dirnames, filenames) in walk(folder):
      websiteFiles.extend(filenames)
      break
  
  print("{} website files recognized.".format(len(websiteFiles)))
  for i in range(len(websiteFiles)):
      filename = websiteFiles[i]
      if i < len(websiteFiles) - 1:
          nextWebsite = websiteFiles[i + 1]
      if i > 0:
          previousWebsite = websiteFiles[i - 1]
      if filename.endswith(".ann"):
          websiteStyle = "annotation"
          if filename.startswith("www."):
              link = filename[:-4]
          else:
              link = "www." + filename[:-4]
          print("{} of {}: {}".format(round(i/2)+1, round(len(websiteFiles)/2),link))
          try:
              websiteData = getSingleWebsiteData(link)
              soup = BeautifulSoup(websiteData, 'html.parser')
              filePath = "{}/{}".format(folder, filename)
              with open(filePath) as f:
                  fileContents = f.read()
                  f.close()
              dataLines = fileContents.splitlines()
              for dataLine in dataLines:
                  if len(dataLine) > 1:
                      dataEntries = dataLine.split("\t")
                      typeOfData = dataEntries[1].split(" ")[0]
                      actualData = dataEntries[2]
                      charsToReplace = ["+", "(", ")"]
                      for character in actualData:
                          if character in charsToReplace:
                              actualData = actualData.replace(
                                  character, "\\" + character)
                      def ReFindall():
                        
                        print("Type of data: {} -   Actual data: {}".format(
                            typeOfData, actualData))
                        searchPattern = '\b(\S+)({})(\S+)\b'.format(actualData)
                        print("Search pattern is: {}".format(searchPattern))
                        searchResult = re.findall(searchPattern, websiteData)
                        if len(actualData) > 0:
                            print(
                                "Sucht man das im WebsiteText direkt, findet man folgendes: {}"
                                .format(re.findall(actualData, websiteData)))
                        print(searchResult)
                      
                      searchPattern = '\b(\S+)({})(\S+)\b'.format(actualData)
                      #title_tag=soup.head.contents[0]
                      #2021 auskommentiert
                      #print(soup.find_all(string=re.compile(searchPattern)))
                      #if "www.baumann-partner.tax" in link:
                      #  print(websiteData)
                      #  break
                      #!!! TODO: Herausfinden, mit welchem Tool Frederic empfahl, in HTML-Code nach Inhalten zu suchen, um den Tag/ das Feature dazu zu finden (um eben diese "actualData" auf der geholten "websiteData" zu finden und auf Tag/ Feature zu schließen. War das BeautifulSoup wie in tryout.py?)
          except:
              print("\tWebsite not found. Moving on to the next...")
                                 
      elif filename.endswith(".txt"):
          websiteStyle = "websitetext"
  print("End of learning algorithm")
  return featureList

#2021-02-21 auskommentiert
#2021-04-13 wieder einkommentiert zum Test
print(learningAlgorithmAnnotatedTexts())
#2021-05-23 featureList ist bisher noch leer!
#2021-05-23 TODO:Die Informationen aus den .ann-Dateien nutzen: Nicht die genauen Positionen bringen mir was, sondern die Klassifizierung der Informationen, z.B. "Tester & partner" als 'Kanzleiname'. Daher diese Informationen dann im HTML-Code finden, den ich oben ja schon pro website selten noch verfügbar abrufe. so bekomme ich die html-features zu diesen Informationen
#2021-10-14 Kann ich es irgendwie hinbekommen, dass ich in learningAlgorithmAnnotatedTexts() nicht nochmal alle webseitentexte herunterladen muss? ansonsten alle gewünschten laden und danach jede einzelne vergleichen mit den .ann-Dateien
#2021-11-01 Debugger verwenden!

# SET TRUE IF NGRAM-APPROACH IS INTENDED
nGramApproach = False
if nGramApproach is True:
    ngrams = first_with_x_count(5, sortDict(nGram(6, websitesTexts)))
    ngramsClean = []
    print("-----------")
    print(ngrams[:20])
    print(similarityOfStrings([ngrams[0], ngrams[1]], str))
    print(similarityOfStrings([ngrams[1], ngrams[2]], str))
    print(similarityOfStrings(['title=', 'itle="'], str))

def dataScraping (websiteText):
  # List of information to contain at least phone number and email adress
  listOfInformation=[(),()]
  
  # 1st round
  

  # 2nd round


  # TODO: WebScraping-Funktion bauen!

  return listOfInformation

dataScraping(websitesTexts[0])

#learningAlgorithmGivenInfo()

#print(websitesTexts)
#print(getMultipleWebsiteData(websitesListDefault))

# TODO: Aktuell werden ja nur die Features gesammelt, daher als nächstes:
# die Features nach Nützlichkeit priorisieren
# auch die Qualität der durch die priorisierten Features eingeholten Informationen (Kanzleiname, Telefonnummer, etc.) bewerten

#TODO (24.08.20): https://www.advocard.de/service/anwaltssuche/ Radius auf 50km von PLZ 40472 aus erhöhen und dann die Liste der Kanzleien erweitern
#TODO (21.09.20): Featureerkennung durch Lerntext funktioniert ja erst mal ganz gut (Name des Dateneintrags im HTML-Code wird ausgelesen, z.B. "faxNumber"). Daher zwei Folgeansätze:
# 1. Selben Lernalgorithmus auf 2, 3 weitere Texte anwenden, deren Daten ich wieder in einer Liste mitliefere
# 2. Den Lernalgorithmus mal bei WebseitenTexten anwenden, für die ich keine Daten mitliefere und überprüfen, ob sich dort nur mit den wenigen bisherig gesammelten zur Auswahl stehenden feature (tags) die wertvollen Informationen des Textes abschöpfen lassen.

#_______
#TODO (14.09.20): Weitere Regex-Suchparameter ausprobieren, um nicht einen ganzen Textblock ('="(.+)"' ), sondern nur die Features zu finden. Dabei beachten: mit regex-Groups durch Klammern lassen sich dann sowohl das Feature als auch die folgende sprachliche Information (um die es ja eigentlich geht) erfassen. Deswegen mit z.B. 2 Groups arbeiten, die eine findet das Feature, die andere die nachstehende Information
#on hold/ vorerst verworfen:
#TODO (14.09.20): Die Lern-/ Trainingsfunktion bauen, um Featureerkennung alternativ zum regelbasierten über Lernansatz hinzubekommen. Dafür Texte unter "trainingTexts.txt" einlesen und Feature-Marker erkennen (°§~)
#TODO (17.09.20): Ab Zeile 171 checken, warum zwar in den Outputs mehrfach "Start of pattern found" steht, aber nie "End of pattern found"
#______
#erledigt (17.09.20): Alternative Lernmethode: Lerntext ohne Annotationen, dafür mit wichtigen Daten wie Telefonnummer an Algorithmus übergeben
