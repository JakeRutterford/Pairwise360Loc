import torch
import torch.nn as nn
import torchvision.models as models

# Full ResNet 34 features, downsample and classify with linear layer
class Model_1(nn.Module):
    def __init__(self):
        super(Model_1, self).__init__()
        # use pretrained feature extractors from resnet - do not train these
        self.features = nn.Sequential(*list(models.resnet34(pretrained=True).children())[:-2])
        for param in self.features.parameters():
            param.requires_grad = False

        # average each feature channel
        self.downsample = nn.AdaptiveAvgPool2d(1)

        # final classifier layers
        self.classifier = nn.Linear(1024, 2)

    def forward(self, image, bubble):
        # extract features using resnet
        image = self.features(image)
        bubble = self.features(bubble)

        # downsample image and bubbles
        image = self.downsample(image)
        bubble = self.downsample(bubble)

        # flatten features
        image = image.view(image.size(0), -1)
        bubble = bubble.view(bubble.size(0), -1)

        # concatenate and run classifier
        output = torch.cat((image, bubble), dim=1)
        output = self.classifier(output)
        return output

# Model 2
class Model_2(nn.Module):
    def __init__(self):
        super(Model_2, self).__init__()
        # use pretrained feature extractors from resnet - do not train these
        self.features = nn.Sequential(*list(models.resnet34(pretrained=True).children())[:-2])
        for param in self.features.parameters():
            param.requires_grad = False

        # average each feature channel
        self.downsample = nn.AdaptiveAvgPool2d(1)

        # final classifier layers
        self.classifier = nn.Sequential(nn.Linear(1024, 512),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(),
                                        nn.Linear(512, 2))

    def forward(self, image, bubble):
        # extract features using resnet
        image = self.features(image)
        bubble = self.features(bubble)

        # downsample image and bubbles
        image = self.downsample(image)
        bubble = self.downsample(bubble)

        # flatten features
        image = image.view(image.size(0), -1)
        bubble = bubble.view(bubble.size(0), -1)

        # concatenate and run classifier
        output = torch.cat((image, bubble), dim=1)
        output = self.classifier(output)
        return output

# Model 3 - Extract features from mid way within the resnet
class Model_3(nn.Module):
    def __init__(self):
        super(Model_3, self).__init__()
        # use pretrained feature extractors from resnet - do not train these
        self.features = nn.Sequential(*list(models.resnet34(pretrained=True).children())[:-4])
        for param in self.features.parameters():
            param.requires_grad = False

        self.image_layer = nn.Sequential(
        nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.AvgPool2d(kernel_size=7, stride=1, padding=0))
        self.bubble_layer = nn.Sequential(
        nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.AvgPool2d(kernel_size=(7,14), stride=1, padding=0))

        # final classifier layers
        self.classifier = nn.Sequential(nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(256, 2))

        #self.classifier = nn.Sequential(
            #nn.Linear(512, 2))

    def forward(self, image, bubble):
        # extract features using pretrained resnet layers
        image = self.features(image)
        bubble = self.features(bubble)

        # run convolution layers
        image = self.image_layer(image)
        bubble = self.bubble_layer(bubble)

        # flatten features
        image = image.view(image.size(0), -1)
        bubble = bubble.view(bubble.size(0), -1)

        # concatenate and run classifier
        output = torch.cat((image, bubble), dim=1)
        output = self.classifier(output)
        return output

# Add the original images as extra features
class Model_4(nn.Module):
    def __init__(self):
        super(Model_4, self).__init__()
        # use pretrained feature extractors from resnet - do not train these
        self.features = nn.Sequential(*list(models.resnet34(pretrained=True).children())[:-4])
        for param in self.features.parameters():
            param.requires_grad = False

        self.downsample_ims = nn.AdaptiveAvgPool2d((28, 28))
        self.downsample_bubs = nn.AdaptiveAvgPool2d((28, 56))

        self.image_layer = nn.Sequential(
        nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.AvgPool2d(kernel_size=7, stride=1, padding=0))
        self.bubble_layer = nn.Sequential(
        nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.AvgPool2d(kernel_size=(7,14), stride=1, padding=0))

        self.classifier = nn.Sequential(nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(256, 2))

        # final classifier layers
        #self.classifier = nn.Sequential(
            #nn.Linear(512, 2))

    def forward(self, image, bubble):
        # downsample images and bubbles for later
        ims = self.downsample_ims(image)
        bubs =  self.downsample_bubs(bubble)

        # extract features using pretrained resnet layers
        image = self.features(image)
        bubble = self.features(bubble)

        # combine resnet features with downsampled versions
        image = torch.cat((image, ims), dim=1)
        bubble = torch.cat((bubble, bubs), dim=1)

        # run convolution layers
        image = self.image_layer(image)
        bubble = self.bubble_layer(bubble)

        # flatten features
        image = image.view(image.size(0), -1)
        bubble = bubble.view(bubble.size(0), -1)

        # concatenate and run classifier
        output = torch.cat((image, bubble), dim=1)
        output = self.classifier(output)
        return output

# Increase the number of convolutional layers
class Model_6(nn.Module):
    def __init__(self):
        super(Model_6, self).__init__()
        # use pretrained feature extractors from resnet - do not train these
        self.features = nn.Sequential(*list(models.resnet34(pretrained=True).children())[:-4])
        for param in self.features.parameters():
            param.requires_grad = False

        self.downsample_ims = nn.AdaptiveAvgPool2d((28, 28))
        self.downsample_bubs = nn.AdaptiveAvgPool2d((28, 56))

        self.image_layer = nn.Sequential(
            nn.Conv2d(131, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.AvgPool2d(kernel_size=4, stride=1, padding=0))

        self.bubble_layer = nn.Sequential(
            nn.Conv2d(131, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.AvgPool2d(kernel_size=(4,7), stride=1, padding=0))

        # final classifier layers
        self.classifier = nn.Sequential(nn.Linear(1024, 2))

    def forward(self, image, bubble):
        # downsample images and bubbles for later
        ims = self.downsample_ims(image)
        bubs =  self.downsample_bubs(bubble)

        # extract features using pretrained resnet layers
        image = self.features(image)
        bubble = self.features(bubble)

        # combine resnet features with downsampled versions
        image = torch.cat((image, ims), dim=1)
        bubble = torch.cat((bubble, bubs), dim=1)

        # run convolution layers
        image = self.image_layer(image)
        bubble = self.bubble_layer(bubble)

        # flatten features
        image = image.view(image.size(0), -1)
        bubble = bubble.view(bubble.size(0), -1)

        # concatenate and run classifier
        output = torch.cat((image, bubble), dim=1)
        output = self.classifier(output)
        return output
