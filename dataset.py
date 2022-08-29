import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
import os
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, data_path:str, transform=None):
        self.data_path = data_path
        if transform is None:
            self.transform = transforms.Compose(
                [
                    # transforms.Resize(size=(224, 224)),
                    transforms.ToTensor(),# convert PIL.Image to tensor, which is GY
                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) # Normalization
                ]
            )
        else:
            self.transform = transform
        self.path_list = os.listdir(data_path)

    def __getitem__(self, idx:int):
        # img to tensor, label to tensor
        img_path = self.path_list[idx]
        abs_img_path = os.path.join(self.data_path, img_path)
        img = Image.open(abs_img_path)
        img = self.transform(img)

        # label = torch.as_tensor(label, dtype=torch.int64) # must use long type, otherwise raise error when training, "expect long"
        return img, abs_img_path, img_path
    def __len__(self) -> int:
        return len(self.path_list)


def dataset_split(full_dataset, train_rate):
    '''
    0.8 0.2
    '''
    train_size = int(len(full_dataset) * train_rate)
    valid_size = (len(full_dataset) - train_size)
    # test_size = valid_size/
    train_set, valid_set = torch.utils.data.random_split(full_dataset, [train_size, valid_size])
    return train_set, valid_set 

def dataloader(dataset, batch_size = 1):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, num_workers=0)
    return data_loader



if __name__ == '__main__':
    batch_size = 20
    train_transform = transforms.Compose([
                  transforms.ToTensor(),
                  transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])

    My_dataset = ImageFolder("./train_dataset", 
                           transform=train_transform, target_transform=None)
    

    train_set, valid_set = dataset_split(My_dataset, 0.8)

    train_loader = dataloader(train_set, batch_size)
    valid_loader = dataloader(valid_set, batch_size)


    for i, item in enumerate(train_loader):
        print(item[0].shape)