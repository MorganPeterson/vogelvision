import torch
import torch.nn as nn

from PIL import Image

from torchvision import models, transforms

from constants import *

def init_vgg16(class_len):
    ''' Create the vgg16 model and init it with our features'''
    vgg16 = models.vgg16_bn()
    vgg16.load_state_dict(
        torch.utils.model_zoo.load_url(PTH))

    # freeze training for all layers
    for param in vgg16.features.parameters():
        param.required_grad = False

    # newly created modules have required_grad=True by default
    num_features = vgg16.classifier[6].in_features
    # remove last layer
    features = list(vgg16.classifier.children())[:-1]
    # add our layer with our number of outputs
    features.extend([nn.Linear(num_features, class_len)])
    # replace the model classifier
    vgg16.classifier = nn.Sequential(*features)

    return vgg16

def init_vgg16_with_model(path, classes, device):
    '''Create a vgg16 model with a pre-trained model'''
    # 'cpu' or 'gpu'
    device = torch.device(device)
    # load weight file
    state_dict = torch.load(path, map_location=device)
    # create vgg16 model
    vgg16 = init_vgg16(classes)
    # load state_dict into vgg16 model
    vgg16.load_state_dict(state_dict)
    # must run eval
    vgg16.eval()

    return vgg16

def img_transformer(img_type, resize=RESIZE, center_crop=CENTER_CROP):
    '''Creates transformers for processing images'''
    if img_type == TRAIN:
        return transforms.Compose([
            # we randomly crop the image to 224x224 and randomly flip
            # horizontally
            transforms.RandomResizedCrop(center_crop),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
            ])
    return transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(center_crop),
        transforms.ToTensor()])

def img_to_tensor(file_path, device):
    '''Given a file path, opens an image and transforms to tensor'''
    image = Image.open(file_path)
    trfms = img_transformer(VAL)
    return trfms(image).to(device).unsqueeze(0)

def img_labels(path):
    '''Opens the classes file and gets label indexes and labels'''
    with open(path) as inf:
        x = inf.readlines()

    return [lines[:-1].split(' ')[1][4:].replace('_', ' ') for lines in x]

