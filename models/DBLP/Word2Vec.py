from gensim.models import Word2Vec

with open('models/DBLP/segmentation.txt', 'r') as file:
    corpus = [line.strip().split() for line in file]

model = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=1, workers=4)

model.save('models/DBLP/word2vec_model.model')
