import torch
import cv2 as cv
from tqdm import tqdm
import torch.nn as nn
from collections import OrderedDict
import torchvision.models as models
from read_yaml import parse_yaml
from dataset import MyDataset, dataloader
from model_create import create_model


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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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