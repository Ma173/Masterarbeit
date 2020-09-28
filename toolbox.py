def sortDict(inputdict):
  listofTuples = sorted(inputdict.items() , reverse=True,  key=lambda x: x[1])
  return listofTuples

def first_with_x_count(minimumCount,listOfTuples):
  exportlist=[]
  for tuple in listOfTuples:
    if tuple[1]>=minimumCount:
      exportlist.append(tuple)
  return exportlist

def loadFromFile(filename):
  loadedFileContents=open(filename,'r')
  return loadedFileContents

def loadListOfTuplesFromFile(filename):
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
      #print("Loaded tuple {}: {}".format(i,(textName,websitetext[:50])))
    if len(parts)==3:
      textName=parts[0]
      websitetext=parts[1]
      finalList.append((textName,websitetext))
      #print("Loaded tuple {}: {}".format(i,(textName,websitetext[:50])))
  print("Loaded {} tuples from file '{}'".format(len(finalList),filename))
  print("First tuple for example is:{}".format((finalList[0][0][:100],finalList[0][1][:50])))
  return finalList

def saveListToFile(listToSave,filename):
  f= open(filename,"w+")
  for i in range(len(listToSave)):
    listElement=listToSave[i]
    if isinstance(listElement,tuple):
      f.write(listElement[0]+"\n")
      f.write(listElement[1]+"\n__________\n")
      print("Saving tuple '{}' to file '{}'.".format(input, filename))
    elif isinstance(listElement,str):
      f.write(listElement+"\n__________\n")
      print("Saving string '{}' to file '{}'.".format(listElement, filename))
    else: print("List contents unknown. Nothing saved to file.")
  f.close()

def similarityOfStrings(listOfItemsToCompare,OutputType):
  from difflib import SequenceMatcher
  import itertools
  if len(listOfItemsToCompare) < 2: return 0.0
  total = sum(SequenceMatcher(None, a, b).ratio() for a, b in itertools.combinations(listOfItemsToCompare, 2))
  cnt = (len(listOfItemsToCompare) * (len(listOfItemsToCompare)-1)) // 2
  if len(listOfItemsToCompare) >0:
    resultCalculation=round(total / cnt,2)
    resultDescription="The similarity of '{}' and '{}' is {}%.".format(listOfItemsToCompare[0],listOfItemsToCompare[1],resultCalculation*100)
  if OutputType is str: return resultDescription
  else: return resultCalculation

def getShortestItem(listToCheck):
  resultString="shortest item: {}".format(min(listToCheck, key=len))
  return resultString