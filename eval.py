from metric import *
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch
from torch import nn

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