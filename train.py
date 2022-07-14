
import  csv
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn,optim
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms
from tqdm import tqdm
from torchvision import datasets
root='./my_data'

transform_train=transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize(128),
     transforms.RandomRotation(10),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
     ])




batch_size=40

train_data = torchvision.datasets.ImageFolder(
    root='./images_train/'
    , transform=transform_train
)

train_loader = torch.utils.data.DataLoader(train_data,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=2)

"""
test_data = torchvision.datasets.ImageFolder(
    root='C:\\Users\\kousu\\ PycharmProjects\\kaonin23\\images\\sprit\\validation'
    , transform=transform_test)

test_data_loader = torch.utils.data.DataLoader(test_data,
                                               batch_size=10,
                                               shuffle=False,
                                               num_workers=2)
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torchvision.models.resnet18(num_classes=4).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

def train(train_loader):
    model.train()
    running_loss=0
    for (images,labels) in tqdm(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_loss = running_loss / len(train_loader)
    return train_loss
"""
def valid(test_loader):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(test_loader)):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            predicted = outputs.max(1, keepdim=True)[1]
            correct += predicted.eq(labels.view_as(predicted)).sum().item() # 先の例で説明済み
            total += labels.size(0)
    val_loss = running_loss / len(test_loader)
    val_acc = correct / total
    return val_loss, val_acc
"""

loss_list = []
val_loss_list = []
val_acc_list = []

num_epochs=50

for epoch in range(num_epochs):
    loss = train(train_loader)
    #val_loss, val_acc = valid(test_loader)
    #print('epoch %d, loss: %.4f val_loss: %.4f val_acc: %.4f' % (epoch, loss, val_loss, val_acc))
    print('epoch %d, loss: %.4f ' % (epoch, loss ))
    loss_list.append(loss)
    ##val_acc_list.append(val_acc)


PATH = './model_train_net2.pth'
torch.save(model.state_dict(), PATH)

#グラフの正答率と全体の正答率が一致した