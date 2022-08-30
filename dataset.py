import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
import os
from PIL import Image


# for testing data's class
class MyDataset(Dataset):
    def __init__(self, data_path:str, transform=None):
        self.data_path = data_path
        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) # Normalization
                ]
            )
        else:
            self.transform = transform
        self.path_list = os.listdir(data_path)

    def __getitem__(self, idx:int):
        # get image's path 
        img_path = self.path_list[idx]
        abs_img_path = os.path.join(self.data_path, img_path)
        img = Image.open(abs_img_path)
        img = self.transform(img)
        return img, abs_img_path, img_path
    def __len__(self) -> int:
        return len(self.path_list)


def dataset_split(full_dataset, train_rate):
    '''
    using random_split to split the whole dataset.

    train 80%
    valid 20%
    '''
    train_size = int(len(full_dataset) * train_rate)
    valid_size = (len(full_dataset) - train_size)
    train_set, valid_set = torch.utils.data.random_split(full_dataset, [train_size, valid_size])
    return train_set, valid_set 

def dataloader(dataset, batch_size):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, num_workers=0)
    return data_loader