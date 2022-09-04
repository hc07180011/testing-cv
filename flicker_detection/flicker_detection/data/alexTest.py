
import numpy as np
test=np.load('./vgg16_emb/0000.mp4.npy',encoding = "latin1")  #加载文件
doc = open('./alexTest/1.txt', 'a')  #打开一个存储文件，并依次写入
print(test, file=doc)  #将打印内容写入文件中

# .npy文件是numpy专用的二进制文件
arr = np.array([[1, 2], [3, 4]])
# 保存.npy文件
np.save("./alexTest/arr.npy", arr)
print("save .npy done")
# # 读取.npy文件
# np.load("./alexTest/arr.npy")
print(test)
print("load .npy done")

