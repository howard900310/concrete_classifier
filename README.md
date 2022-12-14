# concrete_classifier

## requirements
```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip install numpy
pip install matplotlib
pip install torchvision
pip install opencv-python
pip install tqdm
pip install pyyaml
pip install tensorboard
```

### training 使用流程
1. 建立 dataset
2. 更改 `configs.yaml` 內路徑以及 hyper-parameter
3. 執行 `main.py`
4. 得到該次訓練的 accuracy, loss, weight 檔

### testing(分類照片)使用流程
1. 準備好要分類的資料夾，擺放方法是資料夾內直接是所有圖片
![](https://i.imgur.com/VshecqH.png)
2. 執行 `folder_create` 建立輸出圖片目的資料夾
3. 檢查 `configs.yaml` 內路徑
    * `test_path` : 資料夾路徑
    * `test_data_save_path` : 分類後照片放置的路徑
    * `test_weight_path` : 訓練好的 weight 檔路徑
4. 得到結果，手動將分類錯誤的重新分類
5. 執行 `file_name_func.py` 將分類好的照片改名

### testing accuracy(觀察該模型、參數、dataset 是否使效能提升)
1. 將上一個步驟整理好的 dataset 準備好
2. 將 `configs.yaml` 內的 `test_acc_path` 改為第 1 點的資料
3. 執行 `test_acc.py`
4. 得到 accuracy

# 程式碼講解

## 1. dataset 載入, 劃分

* dataset 資料夾擺放方法如下圖，共有 6 個類別
* label 直接由資料夾名稱表示

    ![](https://i.imgur.com/0nXOIgK.png)

* 訓練用的 dataloader 我直接寫在 `main.py` 內，
  1. 由 `ImageFolder` 函數讀取位於 `'train_path'` 的資料夾
  2. 並 import`dataset.py` 內的 `dataset_split`, `dataloader` 兩個 function
  3. 用 `dataset_split` 將讀取到的 dataset 切分為 80% training data, 20% validation data
  4. 用 `dataloader` 將資料讀進 `new_train_loader` 和 `validate_loader` 內
  

```python=
# main.py
from torchvision.datasets import ImageFolder
from torchvision import transforms
from dataset import dataset_split, dataloader

# load training dataset
train_transform = transforms.Compose([
                  transforms.ToTensor(),
                  transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
my_dataset = ImageFolder(train_path, transform=train_transform, target_transform=None)
    # split to 0.8, 0.2
train_set, valid_set = dataset_split(my_dataset, 0.8)
new_train_loader = dataloader(train_set, batch_size)
validate_loader = dataloader(valid_set, batch_size)
```

```python=
# dataset.py
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
```

## 2. network 建立

使用 pytorch 內建的 VGG16 當作 backbone，原本輸出的維度是 1000，所以修改為 6

```python=
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
```
原本 VGG16 架構，將最後一層 (classifier) 替換成自己的使用情形

```python=
# 原本的 VGG16
VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace=True)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace=True)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace=True)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace=True)
    (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): ReLU(inplace=True)
    (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace=True)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace=True)
    (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): ReLU(inplace=True)
    (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (27): ReLU(inplace=True)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace=True)
    (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=1000, bias=True) # 將這行 output_featrue 改為 6
  )
)
```

## 3. metric
* 使用 top-k accuracy 當作評估指標
* 使用 `torch.topk(input, k, dim=None, largest=True, sorted=True, out=None)`
* 求一個 tensor 中前 K 大或前 K 小的 index
> input：一個 tensor 類型的 data
k：指明是得到前k个数据及其index
dim：表示在哪個維度，default 為最後一個維度
largest：True表示從大到小排序，False表示從小到大排序
sorted：True表示返回的结果按照順序返回
out：可省略
```python=
# define top-k accuracy
def accuracy(output, label, topk=(1,)):
    maxk = max(topk)
    batch_size = label.size(0) # 32
    # get top-k index
    _, pred = output.topk(maxk, 1, True, True) # using top-k to get the index of the top k
    pred = pred.t() # transpose
    # eq: compare according to corresponding elements; view(1, -1): automatically converted to the shape of row 1, ;
    # expand_as(pred): shape extended to 'pred'
    # expand_as performs row-by-row replication to expand, and ensure that the columns are equal
    correct = pred.eq(label.view(1, -1).expand_as(pred))
    # print(correct)

    rtn = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0) # flatten the data of first k row to one dim to get total numbers of true
        rtn.append(correct_k.mul_(100.0 / batch_size)) # mul_() tensor's multiplication, (acc num/total num)*100 to become percentage
    return rtn
```

* 建立一個 class 保存更新後的 accuracy
```pthhon=
# using class to save and update accuracy
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
```

## 4. training 函數

```python=
# train.py
from metric import *
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# define train function
def train(train_loader, device, model, epochs, lr, criterion, optimizer, tensorboard_path):
    model = model.to(device) # put model to GPU
    for epoch in range(epochs):
        model.train() # set train mode
        top1 = AverageMeter() # metric
        train_loader = tqdm(train_loader) # convert to tqdm type, convenient to add the output of journal
        train_loss = 0.0
        train_loader.set_description('[%s%04d/%04d %s%f]' % ('Epoch:', epoch+1, epochs, 'lr:', lr))
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device) # put data to GPU
            # initial 0, clear the gradient information of last batch
            optimizer.zero_grad()
            outputs = model(inputs)
            # print('outputs:', outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # calculate topk accuracy
            acc1, acc2 = accuracy(outputs, labels, topk=(1,2))
            n = inputs.size(0) # batch_size
            # print(n)
            top1.update(acc1.item(), n)
            train_loss += loss.item()
            postfix = {'train_loss': '%.6f' % (train_loss / (i + 1)), 'train_acc': '%.6f' % top1.avg}
            train_loader.set_postfix(log=postfix)

            # tensorboard curve drawing
            writer = SummaryWriter(tensorboard_path)
            writer.add_scalar('Train/Loss', loss.item(), epoch)
            writer.add_scalar('Train/Accuracy', top1.avg, epoch)
            writer.flush()
    print('Finished Training')
```

* 有將訓練過程上傳至 tensorboard
* 請訓練完後在 cmd 輸入 `tensorboard --logdir=/path_to_log_dir/ --port 6006` 
即可觀看訓練過程 loss 的變化
![](https://i.imgur.com/vcTs2PF.png)

## 5. evaluation
```python=
# validate.py
from metric import *
from tqdm import tqdm
import torch

def validate(validation_loader, device, model, criterion):

    model = model.to(device) # model --> GPU
    model = model.eval() # set eval mode
    with torch.no_grad():# network does not update gradient during evaluation
        val_top1 = AverageMeter()
        validate_loader = tqdm(validation_loader)
        validate_loss = 0
        for i, data in enumerate(validate_loader):
            inputs, labels = data[0].to(device), data[1].to(device) # data, label --> GPU
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            prec1, prec2 = accuracy(outputs, labels, topk=(1, 2))
            n = inputs.size(0) # batch_size=32
            val_top1.update(prec1.item(), n)
            validate_loss += loss.item()
            postfix = {'validation_loss': '%.6f' % (validate_loss / (i+1)), 'validation_acc': '%.6f'%  val_top1.avg}
            validate_loader.set_postfix(log=postfix)
        val_acc = val_top1.avg
    return val_acc
```


## 6. 建立輸出資料夾

* 再開始訓練前請先執行此程式建立輸出圖片放置的資料夾
* 會建立於自定義的 `output_path` 內

```python=
# model_create.py
import os
from read_yaml import parse_yaml

# create output folder
yaml_path = './configs.yaml'
cfg = parse_yaml(yaml_path)
output_path = cfg['test_data_save_path']

os.mkdir(output_path)
os.mkdir(output_path + '/0_normal')
os.mkdir(output_path + '/1_spalling')
os.mkdir(output_path + '/2')
os.mkdir(output_path + '/3_rebar_exposed')
os.mkdir(output_path + '/4')
os.mkdir(output_path + '/5_unknow')
```



## 7. 主程式(for training)

* 先建立一個 `configs.yaml`放 dataset 的路徑

```python=
# training
train_path: '../dataset/traindata'  # training data 的路徑
tensorboard_path: '../logs/0831_6class' # 存放 logs，方便使用 tensorboard 觀看訓練成果
model_save_path: './weights/weight0831_6class.pth' # 訓練完之 weight 檔儲存路徑

# testing and 分類
test_path: '../dataset/0_24'
test_data_save_path: '../dataset/output'

# test testing accuracy
test_acc_path: '../dataset/output_test'

# hyper-parameter
batch_size: 20
test_batch_size: 1
epochs: 20
lr: 0.001
```

* 主程式

> 這邊需要注意，由於 Loss function 是使用 CrossEntropyLoss，若 dataset 數量除以 `batch_size` 無法整除的話，會發生錯誤，請在建立 training set 時確認好數量分為 80% 20% 後，是否都能整除 `batch_size`。


```python=
# main.py
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

    net = create_model()
    net.load_state_dict(torch.load('./weights/weight0830_6class.pth'))
    # load training dataset
    train_transform = transforms.Compose([
                transforms.ToTensor(),
                # (optional)augmentation
                # transforms.RandomHorizontalFlip(), 
                # transforms.RandomVerticalFlip(),
                # transforms.RandomRotation(degrees=15, expand=False, fill=None),
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
```

## 8. test 分類輸入照片
* 將原始圖片經過訓練好的 network，並透過預測的結果將該圖片放到目標資料夾進行分類
* 由於原始圖片並無分類，所以處理 dataset 的方法與訓練時不一樣
* 透過 `__getitem__`，用 `dataset[idx]` 取出該圖片的絕對路徑(`abs_img_path`) 與檔名(`img_path`)
```python=
# dataset.py
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
```

* 根據輸出結果 `soft_output.argmax(1).item()` 將每張照片按照類別分至資料夾

```python=
# test_classifieer.py
def main():
    # load hyper-parameter
    yaml_path = './configs.yaml'
    cfg = parse_yaml(yaml_path)
    # print(cfg)
    test_path = cfg['test_path']
    test_batch_size = cfg['test_batch_size']
    test_data_save_path = cfg['test_data_save_path']
    test_weight_path = cfg['test_weight_path']

    # load model
    device = torch.device("cuda:0")
    net = create_model()
    net.to(device)
    net.load_state_dict(torch.load(test_weight_path))

    # load testing data
    test_dataset = MyDataset(test_path)
    test_loader = dataloader(test_dataset,test_batch_size)     
    print(len(test_dataset))

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            images = data[0].to(device)
            outputs = net(images)
            # print(data[2][0]) # get image's file name
            softmax_func = nn.Softmax(dim=1) # dim=1 means the sum of rows is 1
            soft_output = softmax_func(outputs) # soft_output is become two probability value

            # print(soft_output.argmax(1).item()) # get the predicted class

            if soft_output.argmax(1).item() == 0: # if class == 0_normal
                test_image = cv.imread(data[1][0]) # read image file
                # cv.imshow('show', test_image)  # display image
                # cv.waitKey()
                cv.imwrite(test_data_save_path + '/0_normal/' + data[2][0] , test_image) # write image to target folder
            elif soft_output.argmax(1).item() == 1: # if class == 1_spalling
                test_image = cv.imread(data[1][0])
                cv.imwrite(test_data_save_path + '/1_spalling/' + data[2][0] , test_image) 
            elif soft_output.argmax(1).item() == 2: # if class == 2
                test_image = cv.imread(data[1][0])
                cv.imwrite(test_data_save_path + '/2/' + data[2][0] , test_image) 
            elif soft_output.argmax(1).item() == 3: # if class == 3_rebar_exposed
                test_image = cv.imread(data[1][0])
                cv.imwrite(test_data_save_path + '/3_rebar_exposed/' + data[2][0] , test_image) 
            elif soft_output.argmax(1).item() == 4: # if class == 4
                test_image = cv.imread(data[1][0])
                cv.imwrite(test_data_save_path + '/4/' + data[2][0] , test_image)
            elif soft_output.argmax(1).item() == 5: # if class == 5_unknow
                test_image = cv.imread(data[1][0])
                cv.imwrite(test_data_save_path + '/5_unknow/' + data[2][0] , test_image)
                
if __name__ == '__main__':
    main()
```

## 9. 測試分類後的效果(accuracy)

* 將 `main.py` 的 validation 部分單獨取出
* 除了路徑外，其他相同

```python=
# test_acc.py
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
```

## 10. 分類後圖片的改名

* 圖片檔名有固定的格式，最後一個數字代表該張圖片的 class 
* e.g. `000318_0_3328_0.jpg`, 該圖片的 `orig_label` 就是 0
* 原本所有圖片的最後一個數字都是 0
* 所以分類完後要根據分類結果更改圖片的名稱
* 此程式是指定圖片的 `orig_label`，並更換為 `new_label`
* `image_folder` 的路徑請指定要變更的照片的根目錄


```python=
# file_name_func.py
import os

def change_file_name(orig_label, new_label, image_folder=str):
    '''
    change the image's label from orig_label to new_label.

    orig_label = 3

    new_label = 1
    
    output : 000007_256_256_3.jpg --> 000007_256_256_1.jpg
    '''

    # find the image's location
    filenames = os.listdir(os.getcwd() + image_folder) 

    data_path = os.getcwd() + image_folder + '\\'

    for name in filenames:
        name_split = name.split(sep = '_')
        # print(name_split)
        if name_split[-1] == str(orig_label) + '.jpg':  # if label is orig_label -> change label
            orig_name = data_path + name
            new_name = data_path + name_split[0]+ '_' + name_split[1] + '_' + name_split[2] + '_' + str(new_label) + '.jpg'
            os.rename(orig_name, new_name)

if __name__ == '__main__':
    change_file_name(orig_label = 1, new_label = 3, image_folder = '\\output_test\\1_spalling')
```
## 11. dataset 分類定義
![](https://i.imgur.com/GCAweKK.png)

![](https://i.imgur.com/qLX1ICu.png)

![](https://i.imgur.com/BTrUo4g.png)

![](https://i.imgur.com/bFunap5.png)
