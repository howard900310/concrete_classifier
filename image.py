from fileinput import filename
import os
from torchvision import transforms
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
from PIL import Image
import cv2 as cv
filenames = os.listdir('./testdata/test')



class MyDataset(Dataset):
    def __init__(self, data_path:str, transform=None):
        self.data_path = data_path
        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(size=(224, 224)),
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
        return img, abs_img_path
    
    def __len__(self) -> int:
        return len(self.path_list)
        



imagess = MyDataset('./testdata/test')
print(imagess[1][1])
test_image = cv.imread(imagess[1][1])
cv.imshow('1', test_image)
cv.waitKey()

