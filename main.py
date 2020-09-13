import requests, re
from tryout import nGram
from toolbox import sortDict, first_with_x_count, loadFromFile,similarityOfStrings, saveListOfTuplesToFile,loadListOfTuplesFromFile, getShortestItem
from chancelleryURLs import websitesDictWithChancelleryName
from userInteraction import userSuggestedFeatures
from itertools import chain
from collections import defaultdict

# Defining all lists of websites to be web scraped

websitesListDefault=["https://www.gansel-rechtsanwaelte.de/","http://www.kanzlei-illner.de/","http://www.hpk-recht.de/","https://www.himmelmann-pohlmann.de/","http://www.anwaltskanzlei-dressler.de/","http://www.anwalt.de/wolff-partner","http://www.advopartner.de/","http://www.anwaelte-mayer.com/","http://www.kanzlei-platanenhof.de/","http://www.rae-teigelack.de/"]
websitesListWithChancelleryName=[["https://www.gansel-rechtsanwaelte.de/","gansel"],["http://www.kanzlei-illner.de/","illner"],["http://www.hpk-recht.de/","heinz"],["https://www.himmelmann-pohlmann.de/","himmelmann"],["http://www.anwaltskanzlei-dressler.de/","dressler"],["http://www.anwalt.de/wolff-partner","wolff"],["http://www.advopartner.de/","advopartner"],["http://www.anwaelte-mayer.com/","mayer"],["http://www.kanzlei-platanenhof.de/","platanenhof"],["http://www.rae-teigelack.de/","teigelack"],["http://www.matyssek-kirchmann.de/","matyssek"],["http://www.rae-drkeller.de/","keller"],["http://www.hp-verteidigung.de/","heimbürger"], ["http://www.sws-rechtsanwaelte.de/","steinhauer"],["http://www.anb-anwaelte.de/","nerger"],["http://www.wittenberg-kollegen.de/","wittenberg"],["http://www.kanzlei-hzk.de/","hering"],["http://www.ra-npp.de/","puhr"],["http://www.kanzlei-duisburg.de/","gärtner"],["http://www.rae-drkeller.de/","keller"]]



websitesListChancelleryComparison1=["https://www.gansel-rechtsanwaelte.de/","http://www.kanzlei-illner.de/"]
websitesListChancelleryComparison2=["http://www.hpk-recht.de/","https://www.gansel-rechtsanwaelte.de/"]
websitesListChancelleryComparison3=["https://www.himmelmann-pohlmann.de/","http://www.hpk-recht.de/"]
websitesListGoogle=["http://www.google.de"]


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
  searchParametersArchive = ['=".*"','="(.+)"','<(.+)>']
  searchParameter3 = '="(.+)"'
  for text in texts:
    foundFeatures = re.findall(searchParameter3,text[1])
    print("Foundall findet {} Treffer".format(len(foundFeatures)))
    # Converting the list of features into a set for removing duplicates easily and then converting it into a list again
    foundFeatures = list(set(foundFeatures))
    print("Text [0] is {}; Text[1][:50] is {}".format(text[0],text[1][:50]))
    websiteName = text[0]
    websitesFeaturesList.append((websiteName,set(foundFeatures)))
    print("\n- {} features in website text '{}' with a length of {}:\n{}\n\n".format(len(foundFeatures),websiteName[:100],len(text[1]),"-"))#foundFeatures))
    #for feature in foundFeatures:
    #  print("\t",feature)
  commonFeatures=[]
  #print("Type of websitesFeaturesList:",type(websitesFeaturesList))
  # Struktur: Liste von 1 Tupel von 2 Strings
  #for item in websitesFeaturesList:
  #  for tupel in item:
  #    print(type(tupel))
  featuresListTexts = []
  
  # Gaining the common features of *all* texts through intersection. Might lead to very few hits
  for textSet in websitesFeaturesList:
    featuresListTexts.append(textSet[1])
    commonFeatures = set.intersection(*featuresListTexts)#websitesFeaturesList[1],websitesFeaturesList[2])
  ##for featureListText in featuresListTexts:
  ##  print("\n\n",featureListText)
  return commonFeatures

# Getting matching features of multiple websites texts by first gathering the websites' texts and then extracting common features

# UNUSED -> ACTIVATE ONCE UPDATING THE WEBSITES' TEXT FILES:
#websitesTexts = getMultipleWebsiteData(websitesDictWithChancelleryName)
# CURRENTLY USED TO LOAD THE WEBSITES' TEXTS FROM FILE:
websitesTexts = loadListOfTuplesFromFile("websitesTexts.txt")

# UNUSED -> ACTIVATE ONCE UPDATING THE WEBSITES' TEXT FILES:
#saveListOfTuplesToFile(websitesTexts,"websitesTexts.txt")

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

userinput=""
loadedFeatures=[]

usereval=False

if usereval==True:
  while userinput!="exit":
    userinput=input("feature: Scan features from the websites' source codes\nload: Load all previously saved features\nexit: Exit the program\n")
    if userinput=="feature":
      userSuggestedFeatures(websitesTexts, websitesListOfFeaturesWithoutWebsitename)
    elif userinput=="load":
      loadedFeatures = loadFromFile("features.txt").readlines()

ngrams=first_with_x_count(5,sortDict(nGram(6,websitesTexts)))
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
ngramsClean=[]
print("-----------")
print(ngrams[:20])
print(similarityOfStrings([ngrams[0],ngrams[1]],str))
print(similarityOfStrings([ngrams[1],ngrams[2]],str))
print(similarityOfStrings(['title=','itle="'],str))

#print(websitesTexts)
#print(getMultipleWebsiteData(websitesListDefault))

# TODO: Aktuell werden ja nur die Features gesammelt, daher als nächstes:
# die Features nach Nützlichkeit priorisieren
# auch die Qualität der durch die priorisierten Features eingeholten Informationen (Kanzleiname, Telefonnummer, etc.) bewerten

#TODO: Features finden über verschiedene Rahmenzeichen
#TODO Die Art überarbeiten wie da Listen und Tupel bei der websitesFeaturesList zusammengemischt werden; die Art die Variable zu bespeichern überarbeiten. Aktuell sind in einer liste (webseite) 4 Tupel, warum auch immer
#TODO (24.08.): Die Outputs (Prints) weiter aufräumen und gucken, warum der Webseiten-Text im Output nicht angezeigt wird (da steht nur '')
#TODO (24.08.): https://www.advocard.de/service/anwaltssuche/ Radius auf 50km von PLZ 40472 aus erhöhen und dann die Liste der Kanzleien erweitern