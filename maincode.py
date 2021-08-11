import os
from sys import api_version
import torch
import numpy as np
import cv2 as cv2
import tqdm as tqdm
from haar import path as haarPath
# st = "00001_script2.jpg"
# print(str.find(st,"_script"))
# if str.find(st,"_script")!=-1:
#     print("yeh")
# else:
#     print("nope")
RE_TRAIN = True

class prepareD():
    IMG_SIZE = 200
    #if you want to re-train the data then change it to True, else False.
    dir = r"C:\Users\ketan\Documents\py\Image rec code\trainset"
    TESTING = r"C:\Users\ketan\Documents\py\Image rec code\trainset\testing"

    list_dir = []
    LABELS = []
    npeye = []
    npe = []
    training_data = []
    count = 1

    def trainImgs(self):
        for i in os.listdir(self.dir):
            self.list_dir.append(i)
            for l in os.listdir(os.path.join(self.dir,i)):
                self.npeye.append(l)
            for j in os.listdir(os.path.join(self.dir,i)):
                for k in os.listdir(os.path.join(os.path.join(self.dir,i),j)):
                    if "_script" in k:
                        self.LABELS.append((k,j))
                    else:
                        pass
        for i in os.listdir(self.dir):
            for j in os.listdir(os.path.join(self.dir,i)):
                for k in os.listdir(os.path.join(os.path.join(self.dir,i),j)):
                    if "_script" in k:
                            # path = os.path.join(os.path.join(os.path.join(os.path.join(dir,i),j)), k)
                            path = f"{self.dir}\\{i}\\{j}\\{k}"
                            cropImg = haarPath(path)
                            # img = cv2.imread(haarPath(path), cv2.IMREAD_GRAYSCALE)
                            img = cv2.resize(cropImg, (self.IMG_SIZE, self.IMG_SIZE))
                            self.count = self.count - 1
                            self.training_data.append([np.array(img), np.eye(len(self.LABELS))[self.count]])  # do something
                            self.npe.append(self.count)
                            # self.npe.append(np.eye(len(self.LABELS))[self.count])
                            self.count = self.count + 1
                        # except Exception as e :
                        #     pass
                    else:
                        pass
                self.count = self.count + 1
        # np.random.shuffle(self.training_data)
        np.save("training_data_4.npy", self.training_data)
        print(len(self.training_data))
        # print(self.npe[1],self.npe[2])


# print(prepareD().trainImgs())

# v = [[1,0,0],[0,1,0],[0,0,5]]
# print(np.eye(len(v))[v[2][2]])

if RE_TRAIN:
        prepareD = prepareD()
        prepareD.trainImgs()
