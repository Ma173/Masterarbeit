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
  finalList=[]
  for item in loadedList:
    #print(item)
    textName=item.split("\n")[0]
    
    finalList.append((textName,item.splitlines()[1]))
  print("Loaded file '{}'".format(filename))
  return finalList

def saveListOfTuplesToFile(listToSave,filename):
  f= open(filename,"w+")
  for listElement in listToSave:
    if isinstance(listElement,tuple):
      f.write(listElement[0]+"\n")
      f.write(listElement[1]+"\n__________\n")
      print("Saving to file '{}'.".format(filename))
    elif isinstance(listElement,str):
      f.write(listElement+"\n__________\n")
      print("Saving to file '{}'.".format(filename))
    else: print("List contents unknown. Nothing saved to file.")
  f.close()

def compare_strings(mylist):
  from difflib import SequenceMatcher
  import itertools
  if len(mylist) < 2: return 0.0
  total = sum(SequenceMatcher(None, a, b).ratio() for a, b in itertools.combinations(mylist, 2))
  cnt = (len(mylist) * (len(mylist)-1)) // 2
  return total / cnt