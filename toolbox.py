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

def saveListToFile(list,filename):
  f= open(filename,"w+")
  for listElement in list:
     f.write(listElement)
     f.write("\n")
  print("Saving to file '{}'.".format(filename))
  f.close()

def compare_strings(mylist):
  from difflib import SequenceMatcher
  import itertools
  if len(mylist) < 2: return 0.0
  total = sum(SequenceMatcher(None, a, b).ratio() for a, b in itertools.combinations(mylist, 2))
  cnt = (len(mylist) * (len(mylist)-1)) // 2
  return total / cnt