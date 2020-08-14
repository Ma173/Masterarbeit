def nGram(n,WebsiteTexts):
  nGramDict={}
  for websitePair in WebsiteTexts:
    websiteName = websitePair[0]
    websiteText = websitePair[1]
    for i in range(len(websiteText)):
      nextN=""
      if i<len(websiteText)-n:
        nextN=websiteText[i:i+n]
        if nextN in nGramDict:
          nGramDict[nextN]+=1
        else: nGramDict[nextN]=1
  return nGramDict
    
