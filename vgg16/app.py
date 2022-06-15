#!/usr/bin/env python

from vgg16 import VGG16

# image directory
data_dir = './data/CUB_200_2011/images'

# set weight_file to None if starting from scratch
weight_file = './birds_half_dataset.pt'

if __name__ == "__main__":
    VGG16(data_dir, weight_file).train_model(num_epochs=1)

