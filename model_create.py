from torchvision import models
from torch import nn
from collections import OrderedDict


def create_model():
    vgg16 = models.vgg16(pretrained=True)
    classifier = nn.Sequential(OrderedDict([('0', nn.Linear(25088, 4096)),
                            ('1', nn.ReLU()), 
                            ('2',nn.Dropout(0.5)),
                            ('3', nn.Linear(4096, 4096)),
                            ('4', nn.ReLU()), 
                            ('5',nn.Dropout(0.5)),
                            ('6', nn.Linear(4096, 6)) # adjust the number of outputs to 6 classes.
                            ]))
    vgg16.classifier = classifier    
    net = vgg16
    return net               