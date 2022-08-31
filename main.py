from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.utils
import torch
from torchvision import models
from torch import nn
from collections import OrderedDict
from torchvision import models
from torch import optim, nn
from train import train
from validate import validate
from read_yaml import parse_yaml
from dataset import dataset_split, dataloader
from model_create import create_model

# main function to be executed
def main():
    # set hyper-parameter of train scripts
    yaml_path = './configs.yaml'
    cfg = parse_yaml(yaml_path)
    # load hyper-parameter
    epochs = cfg['epochs']
    lr = cfg['lr']
    batch_size = cfg['batch_size']
    train_path = cfg['train_path']
    tensorboard_path = cfg['tensorboard_path']
    model_save_path = cfg['model_save_path']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # set model(6 classes)
    # classifier = nn.Sequential(OrderedDict([('0', nn.Linear(25088, 4096)),
    #                       ('1', nn.ReLU()), 
    #                       ('2',nn.Dropout(0.5)),
    #                       ('3', nn.Linear(4096, 4096)),
    #                       ('4', nn.ReLU()), 
    #                       ('5',nn.Dropout(0.5)),
    #                       ('6', nn.Linear(4096, 6))
    #                       ]))
    # vgg16 = models.vgg16(pretrained=True)
    # vgg16.classifier = classifier
    net = create_model()
    net.load_state_dict(torch.load('./weights/weight0830_6class.pth'))
    # load training dataset
    train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(degrees=15, expand=False, fill=None),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
    my_dataset = ImageFolder(train_path, transform=train_transform, target_transform=None)
    # split to 0.8, 0.2
    train_set, valid_set = dataset_split(my_dataset, 0.8)
    new_train_loader = dataloader(train_set, batch_size)
    validate_loader = dataloader(valid_set, batch_size)

    
    criterion = torch.nn.CrossEntropyLoss() # loss function
    optimizer = optim.SGD(net.parameters(), lr, momentum=0.9) # optimizer
    # execute train function
    train(new_train_loader, device, net, epochs, lr, criterion, optimizer, tensorboard_path)
    # save trained model
    torch.save(net.state_dict(), model_save_path)
    # execute validate function
    # 1.load model, load parameter
    val_net = create_model()
    val_net.load_state_dict(torch.load(model_save_path))
    # 2.execute validate function
    validate(validate_loader, device, val_net, criterion)
    print('val_acc:', '%.2f' % validate(validate_loader, device, val_net, criterion) + "%")


if __name__ == '__main__':
    main()