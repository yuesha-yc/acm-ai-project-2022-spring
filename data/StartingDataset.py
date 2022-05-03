import torch
from PIL import Image
import torchvision.transforms as transforms
import os
import pandas as pd
import sys
import numpy as np

from constants import KAGGLE_DATA_PATH, DATA_PATH
from label_generator import id_to_label

data_path = DATA_PATH


class StartingDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).
    """

    # iloc vs loc USE iloc

    def __init__(self, df):
        print(len(df))
        self.images = []

        for i, id in enumerate(df['Image']):
            print(i, id)
            image = Image.open(data_path + id)
            image = image.convert('RGB')
            image = image.resize((224, 224))
            self.images += [transforms.ToTensor()(image)]
            #print(self.images)

        self.labels = df['Id']
        #print("len", len(self.images))
        #print(self.images)

    def __getitem__(self, index):
        
        label = id_to_label(self.labels.iloc[index])
        image_tensor = self.images[index]
        return image_tensor, label
        '''
        inputs = torch.zeros([3, 224, 224])
        label = 0

        return inputs, label
        '''
    def __len__(self):
        return len(self.images)
