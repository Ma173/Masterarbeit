import requests, re
from tryout import nGram
from toolbox import sortDict, first_with_x_count, loadFromFile,compare_strings, saveListOfTuplesToFile

# Defining all lists of websites to be web scraped

websitesListDefault=["https://www.gansel-rechtsanwaelte.de/","http://www.kanzlei-illner.de/","http://www.hpk-recht.de/","https://www.himmelmann-pohlmann.de/","http://www.anwaltskanzlei-dressler.de/","http://www.anwalt.de/wolff-partner","http://www.advopartner.de/","http://www.anwaelte-mayer.com/","http://www.kanzlei-platanenhof.de/","http://www.rae-teigelack.de/"]
websitesListWithChancelleryName=[["https://www.gansel-rechtsanwaelte.de/","gansel"],["http://www.kanzlei-illner.de/","illner"],["http://www.hpk-recht.de/","heinz"],["https://www.himmelmann-pohlmann.de/","himmelmann"],["http://www.anwaltskanzlei-dressler.de/","dressler"],["http://www.anwalt.de/wolff-partner","wolff"],["http://www.advopartner.de/","advopartner"],["http://www.anwaelte-mayer.com/","mayer"],["http://www.kanzlei-platanenhof.de/","platanenhof"],["http://www.rae-teigelack.de/","teigelack"]]
websitesListChancelleryComparison1=["https://www.gansel-rechtsanwaelte.de/","http://www.kanzlei-illner.de/"]
websitesListChancelleryComparison2=["http://www.hpk-recht.de/","https://www.gansel-rechtsanwaelte.de/"]
websitesListChancelleryComparison3=["https://www.himmelmann-pohlmann.de/","http://www.hpk-recht.de/"]
websitesListGoogle=["http://www.google.de"]


# The function to get the text of a single website
def getSingleWebsiteData(url):
  print("\tGathering single website data (text)...")
  websiteText=requests.get(url).text
  print("\tDone gathering data.")
  return websiteText

def getMultipleWebsiteDataWithWebsiteName(urllist):
  print("Gathering text data of {} websites...".format(len(urllist)))
  websitesTexts=[]
  # Iterating through all websites (Website Name & actual url) and saving a list of website name and website text to a list
  for i in range(len(urllist)):
    website =urllist[i]
    print("\n\t{} of {}".format(i,len(urllist)))
    websitesTexts.append((website[1],getSingleWebsiteData(website[0])))
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



def compareWebsitesFeatures(websitesFeaturesList):
  ''' for i in range(len(websitesFeaturesList)):
    currentWebsite=websitesFeaturesList[i]
    currentWebsiteName=currentWebsite[0]
    currentWebsiteFeatures=currentWebsite[1]
    for k in range (len(currentWebsiteFeatures)):
      for l in range '''
from itertools import chain
from collections import defaultdict

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
  print("Extracting common features")
  searchParameterWhole = '=".*"'
  searchParameterLimited = '="(.+)"'
  searchParameter2 = '<(.+)>'
  searchParameter3 = '="(.+)"'
  for text in texts:
    foundFeatures = re.findall(searchParameter3,text[1])
    # Converting the list of features into a set for removing duplicates easily and then converting it into a list again
    foundFeatures = list(set(foundFeatures))
    websiteName = text[0]
    websitesFeaturesList.append((websiteName,set(foundFeatures)))
    print("\n- {} features in website text '{}'".format(len(foundFeatures),websiteName))
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
websitesTexts = getMultipleWebsiteDataWithWebsiteName(websitesListWithChancelleryName[:5])
saveListOfTuplesToFile(websitesTexts,"websitesTexts")

matchingFeatures = textComparisonGetFeatures(websitesTexts)
print("Matching features: ",list(matchingFeatures)[:10])
websitesListOfFeaturesWithoutWebsitename=[]
print("Type of websitesFeaturesList:",type(websitesFeaturesList))
for i in range(len(websitesFeaturesList)):
  print("\t",i,type(websitesFeaturesList[i]))
  #if i==0: print("Erste Liste fängt so an:",websitesFeaturesList[i][:2])
  websitesListOfFeaturesWithoutWebsitename.append(list(websitesFeaturesList[i][1]))

featureFrequency = frequency(websitesListOfFeaturesWithoutWebsitename)
featureFrequencyTop = []
for featurePairs in featureFrequency:
  if featurePairs[1]>1:
    featureFrequencyTop.append(featurePairs)
print("All features that occur more than in just one text:",featureFrequencyTop)

userinput=""
loadedFeatures=[]


def userSuggestedFeatures ():
  userinputFeatures=""
  while userinputFeatures!="exit":
    userinputFeatures=input("\n\nFill in the feature you want to search for. Otherwise type in 'exit' to exit to the program's main loop.\n")
    if userinputFeatures!="exit":
      counterList=[]
      for i in range(len( websitesTexts)):
        currentWebsiteText=websitesTexts[i]
        print("{}: {} matches with the feature you're looking for.\n".format(currentWebsiteText[0],currentWebsiteText[1].count(userinputFeatures)))
        userinputDetail=""
      while userinputDetail!="exit":
        userinputDetail = input("Type in the feature again for a detailed output of found features. Otherwise type in 'exit'.\n")
        if userinputDetail!="exit":
          for i in range(len(websitesListOfFeaturesWithoutWebsitename)):
            currentWebsiteText=websitesTexts[i]
            for feature in websitesListOfFeaturesWithoutWebsitename[i]:
              if re.search(userinputDetail,feature):
                print("{}: This is the full feature occurence you're looking for:{}.\n".format(currentWebsiteText[0],re.findall(userinputDetail,feature)))

usereval=False

if usereval==True:
  while userinput!="exit":
    userinput=input("feature: Scan features from the websites' source codes\nload: Load all previously saved features\nexit: Exit the program\n")
    if userinput=="feature":
      userSuggestedFeatures()
    elif userinput=="load":
      loadedFeatures = loadFromFile("features.txt").readlines()

ngrams=first_with_x_count(5,sortDict(nGram(6,websitesTexts)))
print(compare_strings([ngrams[0],ngrams[1]]))

#print(websitesTexts)
#print(getMultipleWebsiteData(websitesListDefault))

# TODO: Aktuell werden ja nur die Features gesammelt, daher als nächstes:
# die Features nach Nützlichkeit priorisieren
# auch die Qualität der durch die priorisierten Features eingeholten Informationen (Kanzleiname, Telefonnummer, etc.) bewerten

#TODO: Features finden über verschiedene Rahmenzeichen
#TODO Die Art überarbeiten wie da Listen und Tupel bei der websitesFeaturesList zusammengemischt werden; die Art die Variable zu bespeichern überarbeiten. Aktuell sind in einer liste (webseite) 4 Tupel, warum auch immer