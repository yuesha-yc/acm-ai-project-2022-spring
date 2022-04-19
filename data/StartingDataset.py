import torch
from PIL import Image
import torchvision.transforms as transforms
import os
import pandas as pd

DATA_PATH = "data/humpback-whale-identification/train/"


class StartingDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).
    """

    def __init__(self, df): 
        self.images = df['Image']
        self.labels = df['Id']
        print(self.images)

    def __getitem__(self, index):
        print("index: " + str(index))
        image = Image.open(DATA_PATH + self.images[index])
        image_tensor = transforms.ToTensor()(image)
        # TODO: reshape image_tensor
        label = self.labels[index]
        return image_tensor, label
        '''
        inputs = torch.zeros([3, 224, 224])
        label = 0

        return inputs, label
        '''
    def __len__(self):
        return len(self.images)
