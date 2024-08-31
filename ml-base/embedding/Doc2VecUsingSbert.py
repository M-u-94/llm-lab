from sentence_transformers import SentenceTransformer, SentencesDataset, losses
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader
from datasets import Dataset # 不要删除，model.fit 内部需要
from sentence_transformers.util import cos_sim

"""
句向量算法-SBERT

"""

# 初始化预训练的 SBERT 模型
model = SentenceTransformer('all-MiniLM-L6-v2')
# 创建训练数据
train_examples = [
    InputExample(texts=['This is a positive example', 'This is a positive sentence'], label=0.9),
    InputExample(texts=['This is a negative example', 'This is a completely different sentence'], label=0.1)
]

# 将训练数据封装为 SentencesDataset
train_dataset = SentencesDataset(train_examples, model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=2)

# 使用 TripletLoss 进行训练
train_loss = losses.CosineSimilarityLoss(model=model)

# 开始训练
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    warmup_steps=100
)

# 测试句子对的相似性
sentences1 = ['This is an example sentence', 'This is another sentence']
sentences2 = ['This is a different sentence', 'Is this another example?']

# 生成句子嵌入
embeddings1 = model.encode(sentences1, convert_to_tensor=True)
embeddings2 = model.encode(sentences2, convert_to_tensor=True)

# 计算余弦相似度
cosine_scores = cos_sim(embeddings1, embeddings2)

# 输出相似度结果
print("Cosine similarity scores:")
for i in range(len(sentences1)):
    for j in range(len(sentences2)):
        print(f"Sentence 1: '{sentences1[i]}'")
        print(f"Sentence 2: '{sentences2[j]}'")
        print(f"Cosine Similarity Score: {cosine_scores[i][j].item():.4f}")
        print()
