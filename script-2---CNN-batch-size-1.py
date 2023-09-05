import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import time
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import copy
import math
from torchvision.transforms import RandomRotation
from PIL import Image

# fixing random seed, set to True only during development, otherwise you will always get the same results
if False:
    torch.manual_seed(0)
    # random.seed(0)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    # dataloader also may require fixing
torch.autograd.set_detect_anomaly(False)  # set to True only for debugging

# region datasets
train_dataset_50 = datasets.ImageFolder(
    'data/imagenet_50_train/imagenet_images',
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]))

val_dataset_50 = datasets.ImageFolder(
    'data/imagenet_50_val/imagenet_images',
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]))
# endregion

n_epochs = 300
batch_size = 1
learning_rate = 0.005
save = True
load = False
modelName = 'data/script-1---saved-model'

print('batch_size:', batch_size, 'lr:', learning_rate)
print('training dataset length:', len(train_dataset_50))
print('validatin dataset length:', len(val_dataset_50), '\n')

train_dataloader = torch.utils.data.DataLoader(train_dataset_50, batch_size=batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset_50, batch_size=batch_size, shuffle=True)

class Conv2dWS(nn.Conv2d):
    def __init__(self, in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1),
                 padding=(1, 1), dilation=1, groups=1, bias=True):
        super(Conv2dWS, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class NewVGG(nn.Module):
    def __init__(self):
        super(NewVGG, self).__init__()
        self.sequential = nn.Sequential(
            Conv2dWS(3, 64),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            Conv2dWS(64, 64),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            Conv2dWS(64, 128),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            Conv2dWS(128, 128),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            Conv2dWS(128, 256),
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=True),
            Conv2dWS(256, 512),
            nn.GroupNorm(16, 512),
            nn.ReLU(inplace=True),
            Conv2dWS(512, 512),
            nn.GroupNorm(16, 512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            Conv2dWS(512, 512),
            nn.GroupNorm(16, 512),
            nn.ReLU(inplace=True),
            Conv2dWS(512, 512),
            nn.GroupNorm(16, 512),
            nn.ReLU(inplace=True),
            Conv2dWS(512, 512),
            nn.GroupNorm(16, 512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            Conv2dWS(512, 512),
            nn.GroupNorm(16, 512),
            nn.ReLU(inplace=True),
            Conv2dWS(512, 512),
            nn.GroupNorm(16, 512),
            nn.ReLU(inplace=True),
            Conv2dWS(512, 512),
            nn.GroupNorm(16, 512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Flatten(1),
            nn.Linear(in_features=512*7*7, out_features=2048, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=2048, out_features=2048, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_features=2048, out_features=50, bias=True)
        )

    def forward(self, x):
        x = self.sequential(x)
        return x


''' Model training, on 50 classes '''
model = NewVGG().to('cuda:0')

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
""" Loading model """
if load: model.load_state_dict(torch.load(modelName)); print("Model loaded")

print('Epochs number:', n_epochs)
start_time = time.time()
scaler = torch.cuda.amp.GradScaler()  # Automatic Mixed Precision package, torch.cuda.amp, to save memory & time
for epoch in range(0, n_epochs):
    train_loss = 0.0
    valid_loss = 0.0

    start_epoch = time.time()

    """ Training """
    model.train(); print('Training start', '\t %4.1fm' % (time.time() / 60), '\t Epoch:', epoch + 1)

    counter = 0
    batch_multiplier = 20
    for data, target in train_dataloader:
        counter = counter + 1
        data, target = data.cuda(), target.cuda()
        with torch.cuda.amp.autocast():
            output = model(data)
            output.cuda()
        loss = criterion(output, target)
        scaler.scale(loss).backward()
        if counter % 100 == 0:  # just displaying something to show that application works ;)
            print('counter1', counter, '\t %4.1fm' % (time.time()/60))
        if counter % batch_multiplier == 0:  # solution that lowers memory consumption
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        train_loss += loss.item() * data.size(0)
        if counter % 7200 == 0 and save:  # saving model
            torch.save(model.state_dict(), modelName); print("Model saved")

    exp_lr_scheduler.step()

    if save: torch.save(model.state_dict(), modelName); print("Model saved")

    """ Validation """
    model.eval()

    class_correct = list(0. for i in range(200))
    class_total = list(0. for i in range(200))

    good1 = 0
    good5 = 0
    for data, target in val_dataloader:
        with torch.no_grad():
            data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, target)
            valid_loss += loss.item() * data.size(0)
            _, pred = torch.topk(output, 5)
            # compare predictions to true label
            correct_tensor = pred[0, 0].eq(target.data.view_as(pred[0, 0]))
            correct = np.squeeze(correct_tensor.cpu().numpy())
            # calculate validation accuracy for each object class
            if batch_size == 1:
                # top-1 accuracy
                if pred[0, 0].item() == target.data[0].item(): good1 += 1
                # top-5 accuracy
                if pred[0, 0].item() == target.data[0].item()\
                        or pred[0, 1].item() == target.data[0].item()\
                        or pred[0, 2].item() == target.data[0].item()\
                        or pred[0, 3].item() == target.data[0].item()\
                        or pred[0, 4].item() == target.data[0].item(): good5 += 1
            else: print('You need to mplement Accuracy stats for batch_size > 1')

    print('Epoch: {} \nTraining Loss: {:.5f} \nValidation Loss: {:.5f}'.format(
        epoch + 1, train_loss / len(train_dataloader.dataset), valid_loss / len(val_dataloader.dataset)))
    print('Accuracy top-1: %3.1f%% (%2d/%2d)' % (100. * good1 / len(val_dataloader.dataset),
                                           good1, len(val_dataloader.dataset)))
    print('Accuracy top-5: %3.1f%% (%2d/%2d)' % (100. * good5 / len(val_dataloader.dataset),
                                           good5, len(val_dataloader.dataset)))
    print('Epoch time %3.1fm' % ((time.time() - start_epoch)/60), '\n')

print('Total time %3.1fm' % ((time.time() - start_time)/60))
