import requests, re
from tryout import nGram
from toolbox import sortDict, first_with_x_count, loadFromFile,similarityOfStrings, saveListToFile,loadListOfTuplesFromFile, getShortestItem
from chancelleryURLs import websitesDictWithChancelleryName
from userInteraction import userSuggestedFeatures
from itertools import chain
from collections import defaultdict

# The function to get the text of a single website
def getSingleWebsiteData(url):
  print("\tGathering single website data (text)...")
  websiteText=requests.get(url).text
  websiteTextClean=""
  for line in websiteText.splitlines():
    if not line.endswith("</html>"):
      websiteTextClean+=line
    elif line.endswith("</html>"):
      break
  print("\tDone gathering data.")
  return websiteTextClean

def getMultipleWebsiteData(urlCollection):
  urllist=[]
  #print(type(urlCollection))
  if urlCollection is list:
    print("Collection of urls as of list type detected")
    urllist=urlCollection[:]
  elif isinstance(urlCollection,dict):#urlCollection is dict:
    print("Collection of urls as of dictionary type detected")
    for key,value in urlCollection.items():
      pair=[key,value]
      urllist.append(pair)
  print("Gathering text data of {} websites...".format(len(urllist)))
  websitesTexts=[]
  # Iterating through all websites (Website Name & actual url) and saving a list of website name and website text to a list
  for i in range(len(urllist)):
    website =urllist[i]
    #print("website=urllist[i] is: {}".format(urllist[i]))
    print("\n\t{} of {}".format(i,len(urllist)))
    if "http" in website[0]:
      websitesTexts.append((website[1],getSingleWebsiteData(website[0])))
      print("Appending the following: {}".format((website[1],getSingleWebsiteData(website[0])[:50])))
    elif "http" in website[1]:
      websitesTexts.append((website[0],getSingleWebsiteData(website[1])))
      print("Appending the following: {}".format((website[0],getSingleWebsiteData(website[1])[:50])))
  print("Done gathering text data of multiple websites.")
  return websitesTexts

# Getting the chancellery's name
def getChancelleryName(textfile):
  print("Getting chancellery name ...")
  chancelleryNameGroup = re.search('<\/strong><\/h4>\n<h4><strong>(.*)<br \/><\/strong><\/h4', textfile)
  chancelleryName = chancelleryNameGroup.group(1)
  print("Done getting chancellery name.")
  return chancelleryName

overlappingFeaturesDict={}

# Getting the frequency of all elements of multiple lists. Returns tuples of list item and count, sorted by most frequent list item
def frequency(lists): # Notice: If the input is lists (rather than currently a list of lists), insert an asterisk before "lists" -> def frequency(*lists):
    counter = defaultdict(int)
    for x in chain(*lists):
        counter[x] += 1
    return [(key,value) for (key, value) in 
        sorted(counter.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)]

websitesFeaturesList=[]
# Comparing a list of website texts
def textComparisonGetFeatures(texts):
  print("Extracting common features...")
  searchParametersArchive = ['=".*"','="(.+)"','<(.+)>','="(.+)"','(" alt=")(.+)(")']
  searchParameter3 = '(\w+=")([\w\d\s]+)(")'#'(\w+=")(\s*\w+\s*)+(")'
  for text in texts:
    foundFeatures = re.findall(searchParameter3,text[1])
    print("Foundall findet {} Treffer".format(len(foundFeatures)))
    # Converting the list of features into a set for removing duplicates easily and then converting it into a list again
    foundFeatures = list(set(foundFeatures))
    print("Text [0] is {}; Text[1][:50] is {}".format(text[0],text[1][:50]))
    websiteName = text[0]
    websitesFeaturesList.append((websiteName,set(foundFeatures)))
    print("\n- {} features in website text '{}' with a length of {}:\n{}\n\n".format(len(foundFeatures),websiteName[:100],len(text[1]),"---"))#foundFeatures))
    #for feature in foundFeatures:
    #  print("\t",feature)
  commonFeatures=[]
  featuresListTexts = []
  # Gaining the common features of *all* texts through intersection. Might lead to very few hits
  for textSet in websitesFeaturesList:
    featuresListTexts.append(textSet[1])
    commonFeatures = set.intersection(*featuresListTexts)#websitesFeaturesList[1],websitesFeaturesList[2])
  return commonFeatures

# Getting matching features of multiple websites texts by first gathering the websites' texts and then extracting common features

# SWITCH MODE TO UPDATE THE WEBSITES' TEXT FILES:
textImportMode="LoadFromFile"
if textImportMode == "LoadFromFile":
  websitesTexts = loadListOfTuplesFromFile("websitesTexts.txt")
elif textImportMode == "RetrieveFromWeb":
  websitesTexts = getMultipleWebsiteData(websitesDictWithChancelleryName)
  saveListToFile(websitesTexts,"websitesTexts.txt")

matchingFeatures = textComparisonGetFeatures(websitesTexts)
print("Features that match all websites: ",list(matchingFeatures)[:10])
websitesListOfFeaturesWithoutWebsitename=[]
for i in range(len(websitesFeaturesList)):
  #if i==0: print("Erste Liste fängt so an:",websitesFeaturesList[i][:2])
  websitesListOfFeaturesWithoutWebsitename.append(list(websitesFeaturesList[i][1]))

featureFrequency = frequency(websitesListOfFeaturesWithoutWebsitename)
featureFrequencyTop = []
for featurePairs in featureFrequency:
  if featurePairs[1]>2:
    featureFrequencyTop.append(featurePairs)
print("All features that occur on at least 3 websites:")    
for featureFreqPair in featureFrequencyTop:
  print(featureFreqPair)

# SET TRUE TO OFFER USER EVALUATION IN RUNNING CODE
usereval=False
if usereval==True:
  userinput=""
  loadedFeatures=[]

  while userinput!="exit":
    userinput=input("feature: Scan features from the websites' source codes\nload: Load all previously saved features\nexit: Exit the program\n")
    if userinput=="feature":
      userSuggestedFeatures(websitesTexts, websitesListOfFeaturesWithoutWebsitename)
    elif userinput=="load":
      loadedFeatures = loadFromFile("features.txt").readlines()


def countSimilarity(listToCheck):
  tooHighSimilarityCount=0
  for i in range(len(listToCheck)):
    ngramName=listToCheck[i][0]
    for k in range(len(listToCheck)):
      comparedName=listToCheck[k][0]
      similarity=similarityOfStrings([ngramName,comparedName],int)
      if similarity>0.5:
        #print("Similarity between '{}' and '{}' too high ({})!".format(ngramName,comparedName,similarity))
        tooHighSimilarityCount+=1
  print("Too high similarity count:",tooHighSimilarityCount)
  return tooHighSimilarityCount

def learningAlgorithmAnnotatedText(learningTexts):
  ## IMPORT
  LearningTexts_raw = loadFromFile(learningTexts).read()
  learningTextsList_raw = LearningTexts_raw.split("__________")
  learningTextsList = []
  for listItem in learningTextsList_raw:
    if listItem != '':
      learningTextsList.append(listItem)
  print("Imported a list of {} learning texts..".format(len(learningTextsList)))
  print(learningTextsList[0])
  learningtextNameTuples = []
  for textWithName in learningTextsList:
    learningtextNameTuples.append((textWithName.split("\n")[1],textWithName.split("\n")[2]))
  #print(learningtextNameTuples)
  ## ACTUAL LEARNING
  # DELETE THE "[0]" TO LEARN FROM ALL TEXTS
  learnedFeatures=[]
  for i in range(len(learningTextsList)):
    #currentLearningTextTuple = learningtextNameTuples[i]
    currentLearningText = learningTextsList[0]
    learningPattern = "°§~"
    startOfPatternPosition = -1
    endOfFeaturePosition = -1
    endOfPatternPosition = -1
    print("Currently viewed text begins with {}".format(currentLearningText[:50]))
    numberofstartofpatterns=0
    for k in range (len(currentLearningText)):
      currentChar = currentLearningText[k]
      nextChar = ""
      previousChar = ""
      if k<len(currentLearningText)-1: nextChar = currentLearningText[k+1]
      if k>0: previousChar = currentLearningText[k-1]
      
      if currentChar == learningPattern[0] and nextChar == learningPattern[1]:
        print("Start of pattern found.")
        startOfPatternPosition += (k+3)
        numberofstartofpatterns+=1
      elif currentChar == learningPattern[0] and nextChar != learningPattern[1]:
        print("End of pattern found.")
        endOfPatternPosition += (k+3)
        learnedFeatures.append(currentLearningText[startOfPatternPosition:endOfPatternPosition])
        #startOfPatternPosition=0
        #endOfPatternPosition=0
  print("Learning features:\n")
  for feature in learnedFeatures: print(feature)

def learningAlgorithmGivenInfo():
  from learningTextsGivenInfo import websiteTexts as learningTexts
  import re

  # Structure of learningTextsGivenInfo:
  # List 'websiteTexts' = 
      # List of [websiteName, websiteText, websiteData]
      # website Data = List of [(dataName, [list of actual data])]

  featureList=[]
  
  # For each learning text
  for i in range(len(learningTexts)):
    currentTextGroup = learningTexts[i]
    currentTextName = currentTextGroup[0]
    currentActualText = currentTextGroup[1]
    currentTextData = currentTextGroup[2]

    print("\n1) Cycling through each learning text.\nCurrent text: {}".format(currentTextName))
    # For each data entry of the current learning text, e.g.: # phone
    for k in range(len(currentTextData)):
      dataEntry = currentTextData[k]
      dataName = dataEntry[0]
      actualData = dataEntry[1]

      print("\n2) Cycling through each data entry.\nCurrent data entry: {}".format(dataName))

      # For each entity of actual data, e.g.:
      # +4923456789
      # ! The regex search parameter cuts off the search result at whitespace (left and right of the find)
      for l in range (len(actualData)):
        dataEntity = actualData[l]
        searchParameter = '\s(\S*{}\S*)\s'.format(dataEntity)
        searchResults = re.findall(searchParameter,currentActualText)
        print("Current findings of {}: {}".format(dataEntity,searchResults))

        print("\n3) Cycling through each actual data.\nCurrent data: {}".format(dataEntity))

        # For each search result, e.g.:
        # itemprop="telephone">+49 2131 9235-0</span><br
        for m in range(len(searchResults)):
          currentSearchResult = searchResults[m]
          dataPositionInResult = currentSearchResult.find(dataEntity)
          leftOfDataFind = currentSearchResult[:dataPositionInResult]
          rightOfDataFind = currentSearchResult[dataPositionInResult+len(dataEntity):]

          print("\n4) Cycling through each search result.\nCurrent search result: {}".format(currentSearchResult))
          
          if '"' in leftOfDataFind:
            possibleFeature = leftOfDataFind.split('"')[1]
            print("The feature could be {}".format(possibleFeature))
            featureList.append(possibleFeature)
  print("\nFull feature list:{}".format(featureList))
  loadedFeatures = loadFromFile("features.txt").readlines()
  listToSave = []
  listToSave.extend(loadedFeatures)
  listToSave.extend(featureList)
  listToSave = list(set(listToSave))
  saveListToFile(listToSave,"features.txt")


# SET TRUE IF NGRAM-APPROACH IS INTENDED
nGramApproach=False
if nGramApproach is True:
  ngrams=first_with_x_count(5,sortDict(nGram(6,websitesTexts)))
  ngramsClean=[]
  print("-----------")
  print(ngrams[:20])
  print(similarityOfStrings([ngrams[0],ngrams[1]],str))
  print(similarityOfStrings([ngrams[1],ngrams[2]],str))
  print(similarityOfStrings(['title=','itle="'],str))

learningAlgorithmGivenInfo()

#print(websitesTexts)
#print(getMultipleWebsiteData(websitesListDefault))

# TODO: Aktuell werden ja nur die Features gesammelt, daher als nächstes:
# die Features nach Nützlichkeit priorisieren
# auch die Qualität der durch die priorisierten Features eingeholten Informationen (Kanzleiname, Telefonnummer, etc.) bewerten

#TODO (24.08.): https://www.advocard.de/service/anwaltssuche/ Radius auf 50km von PLZ 40472 aus erhöhen und dann die Liste der Kanzleien erweitern
#TODO (21.09.): Featureerkennung durch Lerntext funktioniert ja erst mal ganz gut (Name des Dateneintrags im HTML-Code wird ausgelesen, z.B. "faxNumber"). Daher zwei Folgeansätze:
# 1. Selben Lernalgorithmus auf 2, 3 weitere Texte anwenden, deren Daten ich wieder in einer Liste mitliefere
# 2. Den Lernalgorithmus mal bei WebseitenTexten anwenden, für die ich keine Daten mitliefere und überprüfen, ob sich dort nur mit den wenigen bisherig gesammelten zur Auswahl stehenden feature (tags) die wertvollen Informationen des Textes abschöpfen lassen.

#_______
#TODO (14.09.): Weitere Regex-Suchparameter ausprobieren, um nicht einen ganzen Textblock ('="(.+)"' ), sondern nur die Features zu finden. Dabei beachten: mit regex-Groups durch Klammern lassen sich dann sowohl das Feature als auch die folgende sprachliche Information (um die es ja eigentlich geht) erfassen. Deswegen mit z.B. 2 Groups arbeiten, die eine findet das Feature, die andere die nachstehende Information
#on hold/ vorerst verworfen:
#TODO (14.09.): Die Lern-/ Trainingsfunktion bauen, um Featureerkennung alternativ zum regelbasierten über Lernansatz hinzubekommen. Dafür Texte unter "trainingTexts.txt" einlesen und Feature-Marker erkennen (°§~)
#TODO (17.09.): Ab Zeile 171 checken, warum zwar in den Outputs mehrfach "Start of pattern found" steht, aber nie "End of pattern found"
#______
#erledigt (17.09.): Alternative Lernmethode: Lerntext ohne Annotationen, dafür mit wichtigen Daten wie Telefonnummer an Algorithmus übergeben