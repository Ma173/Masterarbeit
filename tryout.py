def nGram(n,WebsiteTexts):
  nGramDict={}
  for websitePair in WebsiteTexts:
    websiteName = websitePair[0]
    websiteText = websitePair[1]
    for i in range(len(websiteText)):
      char = websiteText[i]
      nextN=""
      if i<len(websiteText)-6:
        nextN=char+websiteText[i+1]+websiteText[i+2]+websiteText[i+3]+websiteText[i+4]+websiteText[i+5]
        if nextN in nGramDict:
          nGramDict[nextN]+=1
        else: nGramDict[nextN]=1
  return nGramDict
    
