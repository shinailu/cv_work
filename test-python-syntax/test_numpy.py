import numpy as np

print(np.exp(-3))
print(np.exp(2))
print(np.exp(-1))
print(np.exp(0))
sum = np.exp(-3)+np.exp(2)+np.exp(-1)+np.exp(0)
sum1 = np.exp(2)+np.exp(-1)+np.exp(0)
print(np.exp(-3)/sum)
print(np.exp(2)/sum)



print(0.3678*2.718)
B = np.arange(3)
print (B)
print (np.exp(B))    # 打印e的幂次方，e是一个常数为2.71828
print (np.sqrt(B))  # 打印B里每个元素的开方
print(B**2)         # 对B求平方

scores = np.array([123, 456, 789])    # example with 3 classes and each having large scores
print(scores)
# scores = np.max(scores) -scores
# print(scores)
scores -= np.max(scores)    # scores becomes [-666, -333, 0]
print(scores)
p = np.exp(scores) / np.sum(np.exp(scores))
print(p)
