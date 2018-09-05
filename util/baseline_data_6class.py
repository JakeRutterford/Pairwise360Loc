import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import PIL
from torch.utils.data.dataset import Dataset
from skimage import io

# custom dataset class
class dataset(Dataset):

    # load dataset
    def __init__(self, dataroot, train=True, augment=True):
        self.images = []
        self.bubbles = []
        self.labels = []

        dataset = [] # contains tuples of images and associated 360 bubbles
        if train:
            file = np.loadtxt(os.path.join(dataroot, "dataset_train.txt"), dtype=str, skiprows=3)
        else:
            file = np.loadtxt(os.path.join(dataroot, "dataset_test.txt"), dtype=str, skiprows=3)

        # load image pairs and create training/validation labels
        for pair in file:
            if 'right' in pair[1]:
                self.labels.extend([i for i in range(3,6)])
            elif 'left' in pair[1]:
                self.labels.extend([i for i in range(0,3)])

            dataset.extend([pair for i in range(3)])

        # calculate dataset length
        self.data_len = len(dataset)

        # transformations when loading images:
        PIL = transforms.ToPILImage()
        resize = transforms.Compose([transforms.Resize(300)])
        bub_size = transforms.Resize(500)
        if augment:
            self.image_trans = transforms.Compose([
                transforms.RandomCrop(224),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.05),
                transforms.ToTensor()])
            self.bubble_trans = transforms.Compose([
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.05),
                transforms.ToTensor()])
        else:
            self.image_trans = transforms.Compose([
                transforms.Resize(300),
                transforms.ToTensor()])
            self.bubble_trans = transforms.Compose([
                transforms.ToTensor()])

        # load images and applying initial preprocessing
        for i, (bubble, image) in enumerate(dataset):
            image = io.imread(os.path.join(dataroot, image), plugin='matplotlib')
            bubble = io.imread(os.path.join(dataroot, bubble), plugin='matplotlib')
            label = torch.tensor(self.labels[i])

            # cropping parameters in height and width, this assumes images of shape (2100, 2800)
            params = [1300, 1300]

            # set left pixel of the image crop depending on label
            if label == 0 or label == 3:
                width = 0
            elif label == 1 or label == 4:
                width = int((image.shape[1] - params[1])/2)
            elif label == 2 or label == 5:
                width = int(image.shape[1] - (params[1] + 1))

            # set height of image crop
            height = int((image.shape[0] - params[0])/2)

            # pre-process image files
            image = PIL(image)
            image = TF.crop(image, height, width, params[0], params[1])
            image = resize(image)
            self.images.append(image)

            # preprocess bubble and add the array
            bubble = PIL(bubble)
            bubble = bub_size(bubble)
            self.bubbles.append(bubble)

    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        # load image, bubble and label
        image = self.images[index]
        bubble = self.bubbles[index]
        label = torch.tensor(self.labels[index])

        # Augment image and bubble and return
        image = self.image_trans(image)
        bubble = self.bubble_trans(bubble)
        return (image, bubble), label

    # Override to give PyTorch size of dataset
    def __len__(self):
        return self.data_len
