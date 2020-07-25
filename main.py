import requests, re
def getWebsiteData(url):
  websiteText=requests.get(url).text
  return websiteText
print(getWebsiteData("http://www.google.de"))