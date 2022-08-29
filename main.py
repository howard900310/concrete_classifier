from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import torch.utils
import torch
from torchvision import models
from torch import nn
from collections import OrderedDict
from torchvision import models
from torch import optim, nn
from train import train
from eval import validate
from read_yaml import parse_yaml
from dataset import dataset_split, dataloader


# main function to be executed
def main():
    # set hyper-parameter of train, eval scripts
    yaml_path = './configs.yaml'
    cfg = parse_yaml(yaml_path)

    epochs = cfg['epochs']
    lr = cfg['lr']
    batch_size = cfg['batch_size']
    # train_path = cfg['train_path']
    # test_path = cfg['test_path']
    tensorboard_path = cfg['tensorboard_path']
    model_save_path = cfg['model_save_path']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    


    classifier = nn.Sequential(OrderedDict([('0', nn.Linear(25088, 4096)),
                          ('1', nn.ReLU()), 
                          ('2',nn.Dropout(0.5)),
                          ('3', nn.Linear(4096, 4096)),
                          ('4', nn.ReLU()), 
                          ('5',nn.Dropout(0.5)),
                          ('6', nn.Linear(4096, 4))
                          ]))

    vgg16 = models.vgg16(pretrained=True)
    vgg16.classifier = classifier
    net = vgg16

    train_transform = transforms.Compose([
                  transforms.ToTensor(),
                  transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])

    my_dataset = ImageFolder("./0_22_cleaned0817", 
                           transform=train_transform, target_transform=None)

    train_set, valid_set = dataset_split(my_dataset, 0.8)

    new_train_loader = dataloader(train_set, batch_size)
    validate_loader = dataloader(valid_set, batch_size)
    
    # train_ds = MyDataset(train_path)
    # new_train_ds, validate_ds = dataset_split(train_ds, 0.8)
    # test_ds = MyDataset(test_path, train=False)

    # new_train_loader = dataloader(new_train_ds, batch_size)
    # validate_loader = dataloader(validate_ds, batch_size)
    # test_loader = dataloader(test_ds, batch_size)

    criterion = torch.nn.CrossEntropyLoss() # loss function
    optimizer = optim.SGD(net.parameters(), lr, momentum=0.9)
    # execute train function
    train(new_train_loader, device, net, epochs, lr, criterion, optimizer, tensorboard_path)
    # save trained model
    torch.save(net.state_dict(), model_save_path)
    # execute evaluation function
    # 1.load model, load parameter
    val_net = vgg16
    val_net.load_state_dict(torch.load(model_save_path))
    # 2.execute evaluation function
    validate(validate_loader, device, val_net, criterion)
    print('val_acc:', '%.2f' % validate(validate_loader, device, val_net, criterion) + "%")
    # 3.generate csv_path
    # submission(csv_path, test_loader, device, net)


if __name__ == '__main__':
    main()