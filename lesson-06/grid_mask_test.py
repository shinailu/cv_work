import matplotlib.pyplot as plt

d1 = 100
d2 = 101
#(d1, d2)定义单元格大小范围

#mask随机旋转角度范围
rotate = 1

#定义mask相对于单元格的比例
ratio = 0.6

mode = 1
prob = 1

grid = Grid(d1, d2, rotate, ratio, mode, prob)
n,c,h,w = img.size()
y = []
for i in range(n):
    y.append(grid(img[i], i==0))
y = torch.cat(y).view(n, c, 425, 425)

plt.imshow(y[0][0], cmap='gray')
plt.axis('off')
plt.show()