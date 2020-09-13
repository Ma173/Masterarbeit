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