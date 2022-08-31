from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.utils
import torch
from torchvision import models
from torch import nn
from torchvision import models
from torch import optim, nn
from train import train
from validate import validate
from read_yaml import parse_yaml
from dataset import dataloader
from model_create import create_model


def main():
    # set hyper-parameter of train scripts
    yaml_path = './configs.yaml'
    cfg = parse_yaml(yaml_path)
    # load hyper-parameter

    test_batch_size = cfg['test_batch_size']
    test_acc_path = cfg['test_acc_path']


    model_save_path = cfg['model_save_path']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    criterion = torch.nn.CrossEntropyLoss() # loss function

    # load training dataset
    train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])

    my_dataset = ImageFolder(test_acc_path, transform=train_transform, target_transform=None)
    test_acc_loader = dataloader(my_dataset, test_batch_size)

    # 1.load model, load parameter
    val_net = create_model()
    val_net.load_state_dict(torch.load(model_save_path))
    # 2.execute validate function
    validate(test_acc_loader, device, val_net, criterion)
    print('val_acc:', '%.2f' % validate(test_acc_loader, device, val_net, criterion) + "%")

if __name__ == '__main__':
    main()