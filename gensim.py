from gensim.test.utils import datapath
from gensim import utils
import gensim.models

class MyCorpus:
  def _iter_(self):
    corpus_path = datapath ('lee_background.cor')
    for line in open(corpus_path):
      yield utils.simple_preprocess(line)

sentences = MyCorpus()
model = gensim.models.Word2Vec(sentences=sentences)

vec_king = model.wv['king']

for index, word in enumerate(wv.index_to_key):
  if index == 10:
    break
  print(f"word #{index}/{len(wv.index_to_key)} is {word}")