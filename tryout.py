from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def nGram(n, WebsiteTexts):
    nGramDict = {}
    for websitePair in WebsiteTexts:
        websiteName = websitePair[0]
        websiteText = websitePair[1]
        for i in range(len(websiteText)):
            nextN = ""
            if i < len(websiteText) - n:
                nextN = websiteText[i:i + n]
                if nextN in nGramDict:
                    nGramDict[nextN] += 1
                else:
                    nGramDict[nextN] = 1
    return nGramDict


def getHtmlData():
    from requests import request
    from bs4 import BeautifulSoup
    link = "https://www.gansel-rechtsanwaelte.de/"
    request = request("GET", url=link)
    soup = BeautifulSoup(request.text, "html5lib")
    subtitle = soup.find_all("title")
    print(type(subtitle[0]))
    print(subtitle[0].get_text())


def gensimSimplePreprocess(text):
    from gensim.utils import simple_preprocess
    print(simple_preprocess(text))


# gensimSimplePreprocess("Der Hund lief durch den Park. Und, man glaubt es kaum, ihm geht es sehr gut!")
def windowsEnvironmentVariable():
    import os
    print(os.environ['PATH'])


def bagOfWordsPlotting():
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    # Liste mit Tupeln, die die Dokumentnamen und Textinhalte enthalten
    documents = [('Dokument 1', 'Dies ist der schöne Textinhalt von Dokument 1'),
                 ('Dokument 2', 'Dies ist der schöne tolle Textinhalt von Dokument 2'),
                 ('Dokument 3', 'Dies ist der gute Textinhalt von Dokument 3')]

    # Erstellen eines leeren Dataframes für die Tf-idf-Werte
    tfidf_df = pd.DataFrame()

    # Initialisierung des TfidfVectorizers und festlegen des Parameters "token_pattern" auf das reguläre Ausdruck-Muster,
    # das nur Adjektive erfasst
    vectorizer = TfidfVectorizer(token_pattern=r'\b\w*(adj)\b')

    # Schleife durch die Dokumente
    for name, text in documents:
        # Transformation des Textinhalts in einen Vektor mit Tf-idf-Werten
        vector = vectorizer.fit_transform([text])

        # Erstellen eines Dataframes für den aktuellen Dokumenten mit den Tf-idf-Werten und den Namen der Adjektive als Spalten
        df = pd.DataFrame(vector.todense(), index=[name], columns=vectorizer.get_feature_names())

        # Anhängen des Dataframes für das aktuelle Dokument an den Gesamt-Dataframe
        tfidf_df = tfidf_df.append(df)

    # Visualisierung der Tf-idf-Werte mit einem horizontalen Balkendiagramm
    tfidf_df.plot(kind='barh', stacked=True)
    plt.show()


def bagOfWordsPlotting2():
    import spacy
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    import matplotlib.pyplot as plt

    # Laden des deutschen Spacy-Modells und Initialisierung des NLP-Pipelines
    nlp = spacy.load('de_core_news_sm')

    # Liste mit Tupeln, die die Kanzlei-Namen und Textinhalte enthalten
    documents = [('Kanzlei 1', 'Wir freuen uns, Sie in unserer tollen Kanzlei Zum goldenen Löwen willkommen zu heißen'),
                 ('Kanzlei 2', 'Schön, wenn Sie uns bald wieder besuchen'),
                 ('Kanzlei 3', 'Unsere Kanzlei liegt im schönen Wuppertal und ist nahe gelegen an der breiten A40')]

    # Initialisierung des TfidfVectorizers
    vectorizer = TfidfVectorizer()

    # Erstellen eines leeren DataFrames für die Tf-idf-Vektoren
    tfidf_df = pd.DataFrame()

    # Schleife durch die Dokumente
    for name, text in documents:
        # Anwenden der NLP-Pipeline auf den Textinhalt
        doc = nlp(text)
        # Tokenisierung und Lemmatisierung des Textinhalts
        tokens = [token.lemma_ for token in doc]
        # Zusammenführen der Tokens zu einem String
        document = ' '.join(tokens)
        # Transformation des Textinhalts in einen Tf-idf-Vektor
        vector = vectorizer.fit_transform([document])
        # Erstellen eines DataFrames für den aktuellen Dokumenten mit den Tf-idf-Werten und den Namen der Adjektive als Spalten
        df = pd.DataFrame(vector.todense(), index=[name], columns=vectorizer.get_feature_names())
        # Anhängen des DataFrames für das aktuelle Dokument an den Gesamt-Dataframe
        tfidf_df = tfidf_df.append(df)

    # Visualisierung der Tf-idf-Werte mit einem horizontalen Balkendiagramm
    tfidf_df.plot(kind='barh', stacked=True)
    plt.show()


bagOfWordsPlotting2()
