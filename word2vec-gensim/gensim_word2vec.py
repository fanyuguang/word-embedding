from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


data_path = 'train_data.txt'
model_path = 'gensim_word2vec.model'
embedding_path = 'train_data_embedding_gensim.txt'

# train
model = Word2Vec(LineSentence(data_path), size=300, window=2, min_count=0, workers=1, sg=0, hs=0, negative=20, sample=1e-4, iter=5)
model.save(model_path)
model.wv.save_word2vec_format(embedding_path, binary=False)

# test
model = Word2Vec.load(model_path)
words = ['知名人士', '美术家', '展览']
for word in words:
  similar_words = model.most_similar(word)
  print('{}, similar words:'.format(word))
  for items in similar_words:
    print('    ({}, {})'.format(items[0], items[1]))
