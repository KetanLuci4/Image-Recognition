import torch
import numpy as np
import cv2 as cv2
import os
import matplotlib.pyplot as plt
from haar import path

training_data = np.load("training_data_1.npy", allow_pickle = True)

X = torch.Tensor([i[0] for i in training_data]).view(-1,175,175)
X = X/255.0
y = torch.Tensor([i[1] for i in training_data])

plt.imshow(X[25], cmap="gray")
plt.show()
print(y[25])

# k = path('C:\\Users\\ketan\\Pictures\\mypic1.jpg')
# l=cv2.imshow("face",k)
# l=cv2.resize(k, (175,175))
# cv2.imshow("resized img",l)
# x = np.array(l)
# print(x)
# k1 = path('C:\\Users\\ketan\\Pictures\\mypic2.jpg')
# l1=cv2.imshow("face",k)
# l1=cv2.resize(k, (175,175))
# cv2.imshow("resized img",l1)
# y = np.array(l1)
# print(y)
# cv2.waitKey(0)
# dir = r"C:\Users\ketan\Documents\py\Image rec code\trainset"
# coun = []
# count = 1
# for i in os.listdir(dir):
#             # list_dir.append(i)
#             # for l in os.listdir(os.path.join(dir,i)):
#             #     # npeye.append(l)
#             for j in os.listdir(os.path.join(dir,i)):
#                 for k in os.listdir(os.path.join(os.path.join(dir,i),j)):
#                     if "_script" in k:
#                         count = count + 1
#                         coun.append(np.eye(2000)[count])
#                         count = count -1
#                     else:
#                         pass
#                     # count = count + 1
#                 count = count + 1


# print(len(coun),coun[1224])
