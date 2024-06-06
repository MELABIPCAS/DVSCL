import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class MEGC2019(torch.utils.data.Dataset):
    """MEGC2019 dataset class with 3 categories"""

    def __init__(self, imgList, transform=None):
        self.imgPath = []
        self.label = []
        self.dbtype = []
        with open(imgList,'r') as f:
            for textline in f:
                texts= textline.strip('\n').split(' ')
                self.imgPath.append(texts[0])
                self.label.append(int(texts[1]))
                self.dbtype.append(int(texts[2]))
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open("".join(self.imgPath[idx]),'r').convert('RGB')
        # plt.imshow(img)
        # plt.show()
        if self.transform is not None:
            img = self.transform(img)
        return img, self.label[idx]

    def __len__(self):
        return len(self.imgPath)

class MEGC2019_SI(torch.utils.data.Dataset):
    """MEGC2019_SI dataset class with 3 categories and other side information"""

    def __init__(self, imgList, transform=None, two_crop=False):
        self.imgPath = []
        self.label = []
        self.dbtype = []
        with open(imgList,'r') as f:
            for textline in f:
                texts= textline.strip('\n').split(' ')
                self.imgPath.append(texts[0])
                self.label.append(int(texts[1]))
                self.dbtype.append(int(texts[2]))
        self.transform = transform
        self.two_crop = two_crop

    def __getitem__(self, idx):
        img = Image.open("".join(self.imgPath[idx]),'r').convert('RGB')
        # plt.imshow(img)
        # plt.show()
        if self.transform is not None:
            img = self.transform(img)
        if self.two_crop:
            img2 = self.transform(img)
            img = torch.cat([img, img2], dim=0)
        return {"data":img, "class_label":self.label[idx], 'db_label':self.dbtype[idx]}

    def __len__(self):
        return len(self.imgPath)


class MEGC2019_SI_rgbd(torch.utils.data.Dataset):
    """MEGC2019_SI dataset class with 3 categories and other side information, 全监督下根据rgb找depth"""

    def __init__(self, imgList, transform=None):
        self.imgPath = []
        self.imgPath1 = []
        self.label = []
        self.dbtype = []
        with open(imgList,'r') as f:
            for textline in f:
                texts= textline.strip('\n').split(' ')
                self.imgPath.append(texts[0])
                self.imgPath1.append(texts[0].replace('casme3_diff_imgs_all', 'casme3_diff_depth_all'))
                self.label.append(int(texts[1]))
                self.dbtype.append(int(texts[2]))
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open("".join(self.imgPath[idx]),'r').convert('RGB')
        img1 = Image.open("".join(self.imgPath1[idx]), 'r').convert('RGB')
        # plt.imshow(img)
        # plt.show()
        if self.transform is not None:
            img = self.transform(img)
            img1 = self.transform(img1)
        # 拼起来
        # 取单通道
        img1 = img1[0, :, :]
        img1 = img1.unsqueeze(0)
        img = torch.cat((img, img1), 0)
        return {"data":img, "class_label":self.label[idx], 'db_label':self.dbtype[idx]}

    def __len__(self):
        return len(self.imgPath)


class MEGC2019_FOLDER(torch.utils.data.Dataset):
    """MEGC2019 dataset class with 3 categories, organized in folders"""

    def __init__(self, rootDir, transform=None):
        labels = os.listdir(rootDir)
        labels.sort()
        self.fileList = []
        self.label = []
        self.imgPath = []
        for subfolder in labels:
            label = []
            imgPath = []
            files = os.listdir(os.path.join(rootDir, subfolder))
            files.sort()
            self.fileList.extend(files)
            label = [int(subfolder) for file in files]
            imgPath = [os.path.join(rootDir, subfolder,file) for file in files]
            self.label.extend(label)
            self.imgPath.extend(imgPath)
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open(self.imgPath[idx],'r').convert('RGB')
        # plt.imshow(img)
        # plt.show()
        if self.transform is not None:
            img = self.transform(img)
        return {"data":img, "class_label":self.label[idx]}

    def __len__(self):
        return len(self.fileList)