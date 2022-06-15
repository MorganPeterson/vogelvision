import copy
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim import lr_scheduler
from torch.autograd import Variable

from torchvision import datasets, models, transforms

from constants import *
from helpers import img_transformer, init_vgg16_with_model, init_vgg16

class VGG16():
    def __init__(self, data_dir, model=None):
        self.data_dir = data_dir
        self.data_transforms = self._create_data_transforms()
        self.img_datasets = self._create_img_datasets()
        self.data_loaders = self._create_data_loaders()
        self.dataset_sizes = self._get_dataset_sizes()
        self.class_names = self._get_class_names()
        self.model_fn = model

        if self.model_fn != None:
            self.vgg16 = init_vgg16_with_model(self.model_fn, len(self.class_names), PYTORCH_DEVICE)
        else:
            self.vgg16 = init_vgg16()

        # define our loss function
        self.criterion = nn.CrossEntropyLoss()
        # define our optimizer
        self.optimizer = optim.SGD(self.vgg16.parameters(), lr=LOSS_RATE, momentum=MOMENTUM)
        # define our scheduler
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    def _create_data_transforms(self):
        '''Takes 224x224 images as input and resize if necessary'''
        return {
                TRAIN: img_transformer(TRAIN),
                VAL: img_transformer(VAL),
                TEST: img_transformer(TEST)
                }

    def _create_img_datasets(self):
        '''Create the image dataset'''
        return {
                x: datasets.ImageFolder(
                    os.path.join(self.data_dir),
                    transform=self.data_transforms[x]
                    )
                for x in [TRAIN, VAL, TEST]
                }

    def _create_data_loaders(self):
        '''Create the loader for the data'''
        return {
                x: torch.utils.data.DataLoader(
                    self.img_datasets[x],
                    batch_size=BATCH_SIZE,
                    shuffle=SHUFFLE,
                    num_workers=NUM_WORKERS
                    )
                for x in [TRAIN, VAL, TEST]
                }

    def _get_dataset_sizes(self):
        '''Get number of images in the dataset'''
        return {x: len(self.img_datasets[x]) for x in [TRAIN, VAL, TEST]}

    def _get_class_names(self):
        '''Get list of class names in dataset'''
        return self.img_datasets[TRAIN].classes

    def train_model(self, num_epochs=DEFAULT_EPOCHS):
        since = time.time()
        best_model_wts = copy.deepcopy(self.vgg16.state_dict())
        best_acc = 0.0

        avg_loss = 0
        avg_acc = 0
        avg_loss_val = 0
        avg_acc_val = 0

        train_batches = len(self.data_loaders[TRAIN])
        val_batches = len(self.data_loaders[VAL])

        for epoch in range(num_epochs):
            print("Epoch {}/{}".format(epoch, num_epochs))
            print('-' * 10)

            loss_train = 0
            loss_val = 0
            acc_train = 0
            acc_val = 0

            self.vgg16.train(True)

            for i, data in enumerate(self.data_loaders[TRAIN]):
                if i % 100 == 0:
                    print("\rTraining batch {}/{}".format(i, train_batches), end='', flush=True)

                inputs, labels = data

                inputs, labels = Variable(inputs), Variable(labels)

                self.optimizer.zero_grad()

                outputs = self.vgg16(inputs)

                _, preds = torch.max(outputs.data, 1)
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()

                loss_train += loss.data
                acc_train += torch.sum(preds == labels.data)

                del inputs, labels, outputs, preds
            print()

            avg_loss = loss_train / self.dataset_sizes[TRAIN]
            avg_acc = acc_train / self.dataset_sizes[TRAIN]

            self.vgg16.train(False)
            self.vgg16.eval()

            with torch.no_grad():
                for i, data in enumerate(self.data_loaders[VAL]):
                    if i % 100 == 0:
                        print("\rValidation batch {}/{}".format(i, val_batches), end='', flush=True)

                    inputs, labels = data

                    self.optimizer.zero_grad()

                    outputs = self.vgg16(inputs)

                    _, preds = torch.max(outputs.data, 1)
                    loss = self.criterion(outputs, labels)

                    loss_val += loss.data
                    acc_val += torch.sum(preds == labels.data)

                    del inputs, labels, outputs, preds

            avg_loss_val = loss_val / self.dataset_sizes[VAL]
            avg_acc_val = acc_val / self.dataset_sizes[VAL]

            print()
            print("Epoch {} result: ".format(epoch))
            print("Avg loss (train): {:.4f}".format(avg_loss))
            print("Avg acc (train): {:.4f}".format(avg_acc))
            print("Avg loss (val): {:.4f}".format(avg_loss_val))
            print("Avg acc (val): {:.4f}".format(avg_acc_val))
            print('-' * 10)
            print()

            if avg_acc_val > best_acc:
                best_acc = avg_acc_val
                best_model_wts = copy.deepcopy(self.vgg16.state_dict())
                torch.save(self.vgg16.state_dict(), self.model_fn)

        elapsed_time = time.time() - since
        print()
        print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
        print("Best acc: {:.4f}".format(best_acc))

        self.vgg16.load_state_dict(best_model_wts)
        torch.save(self.vgg16.state_dict(), self.model_fn)

