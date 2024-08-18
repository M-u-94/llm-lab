"""
独热编码学习
"""
from sklearn.preprocessing import OneHotEncoder

# 原始数据
categories = ["cat", "dog", "rabbit"]

# 转换为2D数组格式（每个样本为一行）
data = [["cat"], ["dog"], ["rabbit"], ["dog"], ["cat"]]

# 创建OneHotEncoder对象
encoder = OneHotEncoder(sparse=False)

# 进行One-Hot编码
one_hot_encoded = encoder.fit_transform(data)
'''
输出结果示例
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]
 [0. 1. 0.]
 [1. 0. 0.]]
'''
print(one_hot_encoded)
