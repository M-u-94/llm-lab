from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# 准备训练数据
sentences = [
    "I love machine learning and natural language processing.",
    "Machine learning models are useful for a variety of tasks.",
    "Natural language processing is a subfield of artificial intelligence.",
    "I enjoy learning new things about machine learning."
]

# 将句子分词
tokenized_sentences = [simple_preprocess(sentence) for sentence in sentences]

# 训练 Word2Vec 模型
model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)

# 获取某个单词的词向量
word_vector = model.wv['machine']
print(f"Vector for 'machine':\n{word_vector}")

# 找到与某个词最相似的词
similar_words = model.wv.most_similar('machine', topn=3)
print("\nMost similar words to 'machine':")
for word, similarity in similar_words:
    print(f"{word}: {similarity:.4f}")
