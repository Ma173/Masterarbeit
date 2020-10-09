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
    
from requests import request
from bs4 import BeautifulSoup
link="https://www.gansel-rechtsanwaelte.de/"
request = request("GET",url=link)
soup = BeautifulSoup(request.text,"html5lib")
subtitle = soup.find_all("title")
print(type(subtitle[0]))
print(subtitle[0].get_text())