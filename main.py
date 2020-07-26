import requests, re, itertools

# Defining all lists of websites to be web scraped

websitesListDefault=["https://www.gansel-rechtsanwaelte.de/","http://www.kanzlei-illner.de/","http://www.hpk-recht.de/","https://www.himmelmann-pohlmann.de/","http://www.anwaltskanzlei-dressler.de/","http://www.anwalt.de/wolff-partner","http://www.advopartner.de/","http://www.anwaelte-mayer.com/","http://www.kanzlei-platanenhof.de/","http://www.rae-teigelack.de/"]
websitesListChancelleryComparison1=["https://www.gansel-rechtsanwaelte.de/","http://www.kanzlei-illner.de/"]
websitesListChancelleryComparison2=["http://www.hpk-recht.de/","https://www.gansel-rechtsanwaelte.de/"]
websitesListChancelleryComparison3=["https://www.himmelmann-pohlmann.de/","http://www.hpk-recht.de/"]
websitesListGoogle=["http://www.google.de"]

# The function to get the text of a single website
def getSingleWebsiteData(url):
  websiteText=requests.get(url).text
  return websiteText

# Getting the texts of multiple websites
def getMultipleWebsiteData(urllist):
  websitesTexts=[]
  for website in urllist:
    websitesTexts.append(getSingleWebsiteData(website))
  return websitesTexts

# Getting the chancellery's name
def getChancelleryName(textfile):
  chancelleryNameGroup = re.search('<\/strong><\/h4>\n<h4><strong>(.*)<br \/><\/strong><\/h4', textfile)
  chancelleryName = chancelleryNameGroup.group(1)
  return chancelleryName

# Comparing a list of website texts
def textComparisonGetFeatures(texts):
  searchParameterWhole = '=".*"'
  searchParameterLimited = '="(.*)"'
  websitesFeaturesList=[]
  for text in texts:
    print()
    websitesFeaturesList.append(set(re.findall(searchParameterLimited,text)))
  commonFeatures=[]
  commonFeatures = set.intersection(*websitesFeaturesList)#websitesFeaturesList[1],websitesFeaturesList[2])
  return commonFeatures

# Getting matching features of multiple websites texts by first gathering the websites' texts and then extracting common features
websitesTexts = getMultipleWebsiteData(websitesListDefault)
matchingFeatures=textComparisonGetFeatures(websitesTexts)
print(list(matchingFeatures)[:10])

#print(getMultipleWebsiteData(websitesListDefault))