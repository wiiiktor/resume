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


def create_dataset(path):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return datasets.ImageFolder(path, transform)

train_dataset = create_dataset('data/imagenet_50_train/imagenet_images')
val_dataset = create_dataset('data/imagenet_50_val/imagenet_images')

batch_size = 1
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

print(f'batch_size: {batch_size} | lr: {0.005}')
print(f'training dataset length: {len(train_dataset)}')
print(f'validation dataset length: {len(val_dataset)}\n')


class Conv2dWS(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(Conv2dWS, self).__init__(*args, **kwargs)

    def forward(self, x):
        weight = self.weight

        # Compute mean and subtract it from the weight
        weight_mean = weight.mean(dim=(1, 2, 3), keepdim=True)
        weight_centralized = weight - weight_mean

        # Compute the standardized weight
        std = weight_centralized.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight_standardized = weight_centralized / std.expand_as(weight)

        return F.conv2d(x, weight_standardized, self.bias, self.stride, self.padding, self.dilation, self.groups)


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

device = torch.device("cuda:0")

model = NewVGG().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
scaler = torch.cuda.amp.GradScaler()

if load:
    model.load_state_dict(torch.load(modelName))
    print("Model loaded")

def train_one_epoch(model, dataloader, optimizer, criterion, scaler, batch_multiplier=20):
    model.train()
    train_loss = 0.0
    counter = 0
    for data, target in dataloader:
        counter += 1
        data, target = data.to(device), target.to(device)
        with torch.cuda.amp.autocast():
            output = model(data)
        loss = criterion(output, target)
        scaler.scale(loss).backward()

        if counter % batch_multiplier == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        train_loss += loss.item() * data.size(0)

        if counter % 7200 == 0 and save:
            torch.save(model.state_dict(), modelName)
            print("Model saved during epoch")

    return train_loss

def validate(model, dataloader, criterion):
    model.eval()
    valid_loss = 0.0
    good1, good5 = 0, 0
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        valid_loss += criterion(output, target).item() * data.size(0)

        _, top5_preds = torch.topk(output, 5)
        correct_tensor = top5_preds[0, 0].eq(target.data.view_as(top5_preds[0, 0]))
        good1 += top5_preds[0, 0].eq(target.data).sum().item()
        good5 += correct_tensor.sum().item()

    return valid_loss, good1, good5

print(f'Epochs number: {n_epochs}')
start_time = time.time()
for epoch in range(n_epochs):
    start_epoch = time.time()

    train_loss = train_one_epoch(model, train_dataloader, optimizer, criterion, scaler)
    valid_loss, good1, good5 = validate(model, val_dataloader, criterion)

    exp_lr_scheduler.step()
    if save:
        torch.save(model.state_dict(), modelName)
        print("Model saved at end of epoch")

    print(f'Epoch: {epoch + 1} \nTraining Loss: {train_loss / len(train_dataloader.dataset):.5f}')
    print(f'Validation Loss: {valid_loss / len(val_dataloader.dataset):.5f}')
    print(f'Accuracy top-1: {100. * good1 / len(val_dataloader.dataset):.1f}% ({good1}/{len(val_dataloader.dataset)})')
    print(f'Accuracy top-5: {100. * good5 / len(val_dataloader.dataset):.1f}% ({good5}/{len(val_dataloader.dataset)})')
    print(f'Epoch time {(time.time() - start_epoch) / 60:.1f}m\n')

print(f'Total time {(time.time() - start_time) / 60:.1f}m')
