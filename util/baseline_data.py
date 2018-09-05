import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from skimage import io

# custom dataset class
class dataset(Dataset):

    # load dataset
    def __init__(self, dataroot, dataset, train=True, augment=True):
        self.dataroot = dataroot
        self.dataset = [] # contains tuples of images and associated 180 bubbles
        self.labels = [] # gives label of whether image is to the left or right of the bubble
        if train:
            file = np.loadtxt(os.path.join(dataroot, "dataset_train" + dataset + ".txt"), dtype=str, skiprows=3)
        else:
            file = np.loadtxt(os.path.join(dataroot, "dataset_test" + dataset + ".txt"), dtype=str, skiprows=3)

        # load image pairs and create training/validation labels
        for pair in file:
            if 'right' in pair[1]:
                label = 0
            elif 'left' in pair[1]:
                label = 1

            # quick fix for images pointing in a different direction to the 360
            if 'backward' in pair[0] and 'forward' in pair[1] or 'forward' in pair[0] and 'backward' in pair[1]:
                if 'right' in pair[1]:
                    label = 1
                elif 'left' in pair[1]:
                    label = 0

            self.dataset.append(pair)
            self.labels.append(label)

        # convert to numpy arrays and calculate dataset length
        self.dataset = np.array(self.dataset)
        self.data_len = len(self.dataset)

        # transformations when loading images
        self.bubble_trans = transforms.Compose([transforms.ToPILImage(),
        transforms.Resize(500), transforms.ToTensor()])

        self.image_trans = transforms.Compose([transforms.ToPILImage(),
        transforms.Resize(300), transforms.ToTensor()])

    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        image = io.imread(os.path.join(self.dataroot, self.dataset[index, 1]), plugin='matplotlib')
        bubble = io.imread(os.path.join(self.dataroot, self.dataset[index, 0]), plugin='matplotlib')

        image = self.image_trans(image)
        bubble = self.bubble_trans(bubble)

        return (image, bubble), torch.tensor(self.labels[index])

    # Override to give PyTorch size of dataset
    def __len__(self):
        return self.data_len
