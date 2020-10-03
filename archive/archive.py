def loadListOfTuplesFromFile_old(filename):
  loadedListFile=open(filename,'r').read()
  loadedList=loadedListFile.split("__________")
  finalList=[]
  for item in loadedList:
    #print(item)
    if item!="":
      textName=item.split("\n")[0]
      textWithoutFirstLine = '\n'.join(item.split('\n')[1:])
      finalList.append((textName,textWithoutFirstLine))
  print("Loaded {} tuples from file '{}'".format(len(finalList),filename))
  print("First tuple for example is:{}".format(finalList[0]))
  return finalList

  def loadListOfTuplesFromFile_old2(filename):
    loadedListFile=open(filename,'r').read()
    loadedList=loadedListFile.split("__________")
    #print("Loaded a list of {} website' texts.".format(len(loadedList)))
    finalList=[]
    for i in range(len(loadedList)):
      item=loadedList[i]
      parts=item.split("\n")
      parts[:] = [x for x in parts if x]
      #print(item)
      if len(parts)==2:
        textName=parts[0]
        websitetext=parts[1]
        finalList.append((textName,websitetext))
        print("Loaded tuple {}: {}".format(i,(textName[:100],websitetext[:50])))
      if len(parts)==3:
        textName=parts[0]
        websitetext=parts[1]
        finalList.append((textName,websitetext))
        print("Loaded tuple {}: {}".format(i,(textName[:100],websitetext[:50])))
    print("Loaded {} tuples from file '{}'".format(len(finalList),filename))
    print("First tuple for example is:{}".format((finalList[0][0][:100],finalList[0][1][:50])))
    return finalList

def learningAlgorithm_old(learningTexts):
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
  for i in range(len(learningtextNameTuples)):
    currentLearningTextTuple = learningtextNameTuples[i]
    currentLearningTextName = currentLearningTextTuple[0]
    currentLearningText = currentLearningTextTuple[1]
    learningPattern = "°§~"
    patternStatus = "no pattern"
    startOfPatternPosition = -1
    endOfFeaturePosition = -1
    endOfPatternPosition = -1
    print("Currently viewed text begins with {}".format(currentLearningText[:50]))
    for k in range (len(currentLearningText)):
      currentChar = currentLearningText[k]
      nextChar = ""
      if k<len(currentLearningText)-1:
        nextChar = currentLearningText[k+1]
      if currentChar == "°":
        if nextChar == "§":
          print("Start of pattern found.")
          startOfPatternPosition += (k+2)
      if patternStatus == "no pattern" and endOfPatternPosition!=-1:
        print("End of pattern found")
        patternStatus = "pattern found"
        learnedFeatures.append(currentLearningText[startOfPatternPosition:endOfPatternPosition])
  print(learnedFeatures)
def learningAlgorithm_old2(learningTexts):
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
    patternStatus = "no pattern"
    startOfPatternPosition = -1
    endOfFeaturePosition = -1
    endOfPatternPosition = -1
    print("Currently viewed text begins with {}".format(currentLearningText[:50]))
    for k in range (len(currentLearningText)):
      currentChar = currentLearningText[k]
      nextChar = ""
      previousChar = ""
      if k<len(currentLearningText)-1:
        nextChar = currentLearningText[k+1]
      if k>0:
        previousChar = currentLearningText[k-1]
      if currentChar == "°" and nextChar == "§":
        print("Start of pattern found.")
        startOfPatternPosition += (k+2)
      elif currentChar == "°" and nextChar != "§":
        print("End of pattern found.")
        endOfPatternPosition += (k+2)
      if patternStatus == "no pattern" and endOfPatternPosition!=-1:
        patternStatus = "pattern found"
        learnedFeatures.append(currentLearningText[startOfPatternPosition:endOfPatternPosition])
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
