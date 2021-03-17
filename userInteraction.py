import re
from toolbox import loadFromFile
def userSuggestedFeatures (websitesTexts,websitesListOfFeaturesWithoutWebsitename):
  userinputFeatures=""
  while userinputFeatures!="exit":
    userinputFeatures=input("\n\nFill in the feature you want to search for. Otherwise type in 'exit' to exit to the program's main loop.\n")
    if userinputFeatures!="exit":
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

def userinput(websitesTexts,websitesListOfFeaturesWithoutWebsitename):
  userinput = ""
  loadedFeatures = []

  while userinput != "exit":
      userinput = input(
          "feature: Scan features from the websites' source codes\nload: Load all previously saved features\nexit: Exit the program\n"
      )
      if userinput == "feature":
          userSuggestedFeatures(websitesTexts,
                                websitesListOfFeaturesWithoutWebsitename)
      elif userinput == "load":
          loadedFeatures = loadFromFile("features.txt").readlines()