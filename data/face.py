import torch.utils.data as data
import csv
from PIL import Image
from random import shuffle
from math import ceil


class FaceDatasets(data.Dataset):
    def __init__(self, csv_path, train=True, transform=None) -> None:
        self.train = train
        self.transform = transform
        with open(csv_path, 'r') as csv_file:
            facereader = csv.reader(csv_file)
            self.image_age_list = list(facereader)

        shuffle(self.image_age_list)
        portion = ceil(len(self.image_age_list) * 0.7)
        self.train_list = self.image_age_list[:portion]
        self.test_list = self.image_age_list[portion:]

    def __len__(self):
        if self.train:
            return len(self.train_list)
        else:
            return len(self.test_list)

    def __getitem__(self, index):
        if self.train:
            img = Image.open(self.train_list[index][0])
            age = self.train_list[index][1]
        else:
            img = Image.open(self.test_list[index][0])
            age = self.test_list[index][1]
        img = self.transform(img)
        return img, age


if __name__ == '__main__':
    datasets = FaceDatasets('/home/gdshen/datasets/face/processed/imdb.csv')
