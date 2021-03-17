from itertools import chain
from collections import defaultdict

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

# Write a list to a file and overwrite any existing data in file
def saveListToFile(listToSave,filename):
  f= open(filename,"w+")
  for i in range(len(listToSave)):
    listElement=listToSave[i]
    if listElement=="" or listElement==" " or listElement=="\n":
      continue
    if isinstance(listElement,tuple):
      f.write(listElement[0]+"\n")
      f.write(listElement[1]+"\n__________\n")
    elif isinstance(listElement,str):
      f.write(listElement+"\n")#+"\n__________\n")
    else: print("List contents unknown. List element was {}Nothing saved to file.".format(type(listElement)))
  print("Saved a list of {} items to file '{}'.".format(len(listToSave), filename))
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

# Getting the frequency of all elements of multiple lists. Returns tuples of list item and count, sorted by most frequent list item
def frequency(lists):  # Notice: If the input is lists (rather than currently a list of lists), insert an asterisk before "lists" -> def frequency(*lists):
    counter = defaultdict(int)
    for x in chain(*lists):
        counter[x] += 1
    return [(key, value) for (key, value) in sorted(
        counter.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)]

def countSimilarity(listToCheck):
    tooHighSimilarityCount = 0
    for i in range(len(listToCheck)):
        ngramName = listToCheck[i][0]
        for k in range(len(listToCheck)):
            comparedName = listToCheck[k][0]
            similarity = similarityOfStrings([ngramName, comparedName], int)
            if similarity > 0.5:
                #print("Similarity between '{}' and '{}' too high ({})!".format(ngramName,comparedName,similarity))
                tooHighSimilarityCount += 1
    print("Too high similarity count:", tooHighSimilarityCount)
    return tooHighSimilarityCount