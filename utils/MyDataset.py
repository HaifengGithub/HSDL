from PIL import Image
import torch
from torch.utils.data import Dataset
class ToolnetDataset(Dataset):
    def __init__(self, txt_path, transform = None, target_transform = None):
        fh = open(txt_path, 'r', encoding='utf8')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split(',')
            imgs.append((words[0], words[1:]))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
    def __getitem__(self, index):
        fn, label = self.imgs[index]
        # print(fn)
        img = Image.open(fn).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label
    def __len__(self):
        return len(self.imgs)

class EndonetDataset(Dataset):
    def __init__(self, txt_path, transform = None, target_transform = None):
        fh = open(txt_path, 'r', encoding='utf8')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split(',')
            imgs.append((words[0], words[1:]))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
    def __getitem__(self, index):
        fn, label = self.imgs[index]
        # print(fn)
        img = Image.open(fn).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label
    def __len__(self):
        return len(self.imgs)

class LSTMDataset(Dataset):
    def __init__(self, txt_path, transform = None, target_transform = None):
        fh = open(txt_path, 'r', encoding='utf8')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split(';')
            imgs.append((words[0], words[1]))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.default_imgs1 = imgs[0]
        self.default_imgs2 = imgs[-1]
    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img_string = Image.open(fn.split(',')[0]).convert('RGB')
        if self.transform is not None:
            img_string = self.transform(img_string)
        img_string = img_string.unsqueeze(0)
        # print(img_string.shape)
        for fn1 in fn.split(',')[1:]:
            img = Image.open(fn1).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            img_string = torch.cat((img_string, img.unsqueeze(0)), 0)
            # print(img_string.shape)
        return img_string, label.split(',')
    def __len__(self):
        return len(self.imgs)

class TinytoolnetDataset(Dataset):
    def __init__(self, txt_path, transform = None, target_transform = None):
        fh = open(txt_path, 'r', encoding='utf8')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split(',')
            imgs.append((words[0], int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
    def __getitem__(self, index):
        fn, label = self.imgs[index]
        # print(fn)
        img = Image.open(fn).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label
    def __len__(self):
        return len(self.imgs)