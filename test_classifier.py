import torch
import cv2 as cv
from tqdm import tqdm
import torch.nn as nn
from collections import OrderedDict
import torchvision.models as models
from read_yaml import parse_yaml
from dataset import MyDataset, dataloader



# this function is only for debugging
def main():
    yaml_path = './configs.yaml'
    cfg = parse_yaml(yaml_path)
    print(cfg)
    test_path = cfg['test_path']
    test_data_save_path = cfg['test_data_save_path']
    # batch_size = cfg['batch_size']

    # load model
    classifier = nn.Sequential(OrderedDict([('0', nn.Linear(25088, 4096)),
                          ('1', nn.ReLU()), 
                          ('2',nn.Dropout(0.5)),
                          ('3', nn.Linear(4096, 4096)),
                          ('4', nn.ReLU()), 
                          ('5',nn.Dropout(0.5)),
                          ('6', nn.Linear(4096, 6))
                          ]))
    device = torch.device("cuda:0")
    net = models.vgg16(pretrained=False)
    net.classifier = classifier
    net.to(device)
    net.load_state_dict(torch.load('./weights/weight0830_6class.pth'))

    test_dataset = MyDataset(test_path)
    test_loader = dataloader(test_dataset)     
    print(len(test_dataset))

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            images = data[0].to(device)
            outputs = net(images)
            # print(data[2][0]) # 取出圖片檔名
            softmax_func = nn.Softmax(dim=1) # dim=1 means the sum of rows is 1
            soft_output = softmax_func(outputs) # soft_output is become two probability value

            # print(soft_output.argmax(1).item())

            if soft_output.argmax(1).item() == 0:
                test_image = cv.imread(data[1][0])
                # cv.imshow('show', test_image)
                # cv.waitKey()
                cv.imwrite(test_data_save_path + '/0_normal/' + data[2][0] , test_image)
            elif soft_output.argmax(1).item() == 1:
                test_image = cv.imread(data[1][0])
                cv.imwrite(test_data_save_path + '/1_spalling/' + data[2][0] , test_image)
            elif soft_output.argmax(1).item() == 2:
                test_image = cv.imread(data[1][0])
                cv.imwrite(test_data_save_path + '/2/' + data[2][0] , test_image)
            elif soft_output.argmax(1).item() == 3:
                test_image = cv.imread(data[1][0])
                cv.imwrite(test_data_save_path + '/3_rebar_exposed/' + data[2][0] , test_image)
            elif soft_output.argmax(1).item() == 4:
                test_image = cv.imread(data[1][0])
                cv.imwrite(test_data_save_path + '/4/' + data[2][0] , test_image)
            elif soft_output.argmax(1).item() == 5:
                test_image = cv.imread(data[1][0])
                cv.imwrite(test_data_save_path + '/5_unknow/' + data[2][0] , test_image)
                
 
if __name__ == '__main__':
    main()