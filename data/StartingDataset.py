import torch
from PIL import Image
import torchvision.transforms as transforms
import os
import pandas as pd
import sys

DATA_PATH = "data/humpback-whale-identification/train/"


class StartingDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).
    """

    #iloc vs loc USE iloc

    def __init__(self, df): 
        self.images = df['Image']
        self.labels = df['Id']
        #print(self.images)
        #print(type(self.images))
        #print(self.labels)

    def __getitem__(self, index):
        print("index: " + str(index))
        try:
            id = self.images[index]
            self.images.to_csv('dump.csv')
        except:
            #sys.exit('failed to open')
            print(f"Failed to open {index}") 
            id = "fffde072b.jpg"
        image = Image.open(DATA_PATH + id)
        image = image.convert('RGB')
        image = image.resize((600, 1050))

        image_tensor = transforms.ToTensor()(image)
        print(image_tensor.shape)

        label = self.labels[index]
        return image_tensor, label
        '''
        inputs = torch.zeros([3, 224, 224])
        label = 0

        return inputs, label
        '''
    def __len__(self):
        return len(self.images)
