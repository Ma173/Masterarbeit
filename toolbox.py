def sortDict(inputdict):
  listofTuples = sorted(inputdict.items() , reverse=True,  key=lambda x: x[1])
  return listofTuples

def firstX(minimumCount,listOfTuples):
  exportlist=[]
  for tuple in listOfTuples:
    if tuple[1]>=minimumCount:
      exportlist.append(tuple)
  return exportlist

def loadFile(filename):
  loadedFileContents=open(filename,'r')
  return loadedFileContents
def saveFile(filename):
  contentToSave = open(filename, 'w')