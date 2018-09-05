# pytorch modules used
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

# dataloader and models
from util.data_loader import dataset
from util import base_model

# other modules used
import argparse
import numpy as np
import pickle
import time
import copy
import os

def params():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataroot', type=str, default='outdoor', help='Dataset File Path')
    parser.add_argument('--model', type=int, default=1, help='Training Model')
    parser.add_argument('--dataset', type=str, default='', help='Dataset To Use')
    parser.add_argument('--epochs', type=int, default=200, help='Number of Training Epochs')
    parser.add_argument('--gpu', type=int, default=0, help='GPU to Use if Available')
    parser.add_argument('--info', type=bool, default=True, help='Save Training/Validation Info')
    parser.add_argument('--params', type=bool, default=False, help='Save Best Model Weights')
    return parser.parse_args()

# global parameters
args = params()

# if GPU available use
device = "cuda:" + str(args.gpu)
device = torch.device(device if torch.cuda.is_available() else "cpu")

def main():
    # load images from dataroot
    print("Loading images...")
    image_datasets = {x: dataset(args.dataroot, train=i, augment=i, dataset=args.dataset) for (x, i) in [('Train', True), ('Val', False)]}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=8, shuffle=True) for x in ['Train', 'Val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['Train', 'Val']}

    # Initialise model - default model is ResNet18 but others have been tested
    if args.model == 1:
        model = base_model.Model_1()
    elif args.model == 2:
        model = base_model.Model_2()
    elif args.model == 3:
        model = base_model.Model_3()
    elif args.model == 4:
        model = base_model.Model_4()
    elif args.model == 6:
        model = base_model.Model_6()
    model.to(device)

    # set training criterion and optimizer - only use parameters with grad required true
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(list(filter(lambda p: p.requires_grad, model.parameters())), lr=0.1, momentum=0.5)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    print("Training on " + str(device) + "...")
    model, results = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=args.epochs)

    # save training results to csv
    if args.info:
        print("Saving Training/Validation Information...")
        results = np.asarray(results)
        np.savetxt(os.path.join("model_output/info_" + str(args.model) + ".csv"), results, delimiter=",")

    if args.params:
        print("Saving Model Weights...")
        # need to write the save model weights function

    print("Finished")

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs):
    since = time.time()
    results = []

    # store weights of model that performs best on validation dataset
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['Train', 'Val']:
            if phase == 'Train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_accuracy = 0.0

            # Iterate over data.
            for (image, bubble), labels in dataloaders[phase]:
                image = image.to(device)
                bubble = bubble.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward - track history if only in train
                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(image, bubble)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()

                # running statistics
                running_loss += loss.item() * image.size(0)
                running_accuracy += torch.sum(preds == labels.data).item()

            # epoch statistics
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_accuracy / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # store training stats if in train mode
            if phase == 'Train':
                train_loss = epoch_loss
                train_acc = epoch_acc

            # store val stats if in val, also copy model if improvement
            if phase == 'Val':
                val_loss = epoch_loss
                val_acc = epoch_acc
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        # append results
        results.append([epoch, train_loss, train_acc, val_loss, val_acc])
        print()

    # report training statistics
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best Val Acc: {:4f}'.format(best_acc))

    # return best model and results
    model.load_state_dict(best_model_wts)
    return model, results

if __name__=="__main__":
    main()
