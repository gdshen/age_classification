import csv
import os
from random import shuffle

import torch.utils.data as data
from PIL import Image


class IMDBWIKIDatasets(data.Dataset):
    def __init__(self, csv_path, train=True, transform=None, target_transform=None) -> None:
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        # csv content sample
        # /home/gdshen/datasets/face/imdb_crop/01/nm0000001_rm124825600_1899-5-10_1968.jpg,69
        # /home/gdshen/datasets/face/imdb_crop/01/nm0000001_rm3343756032_1899-5-10_1970.jpg,71
        # /home/gdshen/datasets/face/imdb_crop/01/nm0000001_rm577153792_1899-5-10_1968.jpg,69
        with open(csv_path, 'r') as csv_file:
            facereader = csv.reader(csv_file)
            self.image_age_list = [[row[0], int(row[1])] for row in
                                   list(facereader)]  # age read from csv is in type str, should convert to int

        shuffle(self.image_age_list)

    def __len__(self):
        return len(self.image_age_list)

    def __getitem__(self, index):
        img = Image.open(self.image_age_list[index][0]).convert(
            'RGB')  # 4 hours debug, find that there are grayscale images, not all images are rgb
        age = self.image_age_list[index][1]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            age = self.target_transform(age)
        return img, age


class AsianFaceDatasets(data.Dataset):
    def __init__(self, csv_path, img_dir, train=True, transform=None, target_transform=None):
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        # csv content sample
        # File,Gender,Age
        # D:\work\mengmeng\data\yueda889\frontal\001614d6-9198-4f14-bb34-1b0696587902.jpg,Female,26
        # D:\work\mengmeng\data\yueda889\frontal\001ab0e7-2b15-45a7-8733-2bae6388357a.jpg,Female,26
        # D:\work\mengmeng\data\yueda889\frontal\001c0e3b-1dca-4021-9be7-3edb7357895b.jpg,Male,22
        with open(csv_path, 'r') as csv_file:
            facereader = csv.reader(csv_file)
            facereader = list(facereader)[1:]
            self.img_age_list = [(os.path.join(img_dir, row[0].split('/')[-1]), int(row[-1])) for row in facereader]
        if self.train:
            shuffle(self.img_age_list)

    def __len__(self):
        return len(self.img_age_list)

    def __getitem__(self, index):
        img = Image.open(self.img_age_list[index][0])  # all images are in rgb mode
        age = self.img_age_list[index][1]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            age = self.target_transform(age)
        return img, age


class WholeFaceDatasets(data.Dataset):
    def __init__(self, csv_path, img_base_dir, train=True, transform=None):
        self.train = train
        self.transform = transform
        with open(csv_path) as csv_file:
            facereader = csv.reader(csv_file)
            facereader = list(facereader)
            self.img_age_list = [(os.path.join(img_base_dir, row[0]), int(row[1])) for row in facereader]

        if self.train:
            shuffle(self.img_age_list)

    def __len__(self):
        return len(self.img_age_list)

    def __getitem__(self, index):
        img = Image.open(self.img_age_list[index][0])
        age = self.img_age_list[index][1]
        if self.transform is not None:
            img = self.transform(img)
        return img, age


if __name__ == '__main__':
    datasets = IMDBWIKIDatasets('/home/gdshen/datasets/face/processed/imdb.csv')
