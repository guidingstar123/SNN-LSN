from __future__ import print_function
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
import time
from torch.nn import DataParallel
from lsn import *
from spikingjelly.clock_driven import neuron, functional, surrogate

os.environ['CUDA_VISIBLE_DEVICES'] = "2"
names = 'lsn101'
data_path = '/home/guidingstar/datasets/cifar10'  # todo: input your data path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
learning_rate = 5e-3
epochs = 300
tau = 2.0
thresh = 0.5
step_size = 10

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
acc_record = list([])
loss_train_record = list([])
loss_test_record = list([])

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

train_dataset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=train_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=False)

test_set = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=test_transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=False)


scalar = torch.cuda.amp.GradScaler()
    
model = lsn101()
model.to(device)
model = DataParallel(model)
criterion = nn.CrossEntropyLoss()
# 0.005
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.005)
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.005)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5, last_epoch=-1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300, eta_min=0)

for epoch in range(epochs):
    model.train()
    correct = 0
    total = 0
    running_loss = 0
    start_time = time.time()
    for i, (images, labels) in enumerate(train_loader):
        model.zero_grad()
        optimizer.zero_grad()

        labels = labels.to(device)
        images = images.to(device)
        with torch.cuda.amp.autocast():
            outputs = model(images)
            #outputs = outputs.mean(1)
            loss = criterion(outputs, labels.long())
        running_loss += loss.item()
        pred = outputs.argmax(dim=1)  # 返回每一行中最大值元素索引
        total += images.size(0)
        correct += torch.eq(pred, labels).sum().item()
        #loss.backward()
        #optimizer.step()
        scalar.scale(loss).backward()
        scalar.step(optimizer)
        scalar.update()
        functional.reset_net(model)
        if (i + 1) % 600 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.5f'
                  % (epoch + 1, epochs, i + 1, len(train_dataset) // batch_size, running_loss))
            running_loss = 0
            print('Time elasped:', time.time() - start_time)
    acc = 100. * float(correct) / float(total)
    print(' Acc: %.5f' % acc)
    correct = 0
    total = 0
    scheduler.step()
    with torch.no_grad():
        model.eval()
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            pred = outputs.argmax(dim=1)
            total += inputs.size(0)
            correct += torch.eq(pred, targets).sum().item()
            functional.reset_net(model)
    print('Iters:', epoch, '\n\n')
    print('Test Accuracy of the model on the 10000 test images: %.3f' % (100 * correct / total))
    #if 100 * correct / total > best_acc:
        #torch.save(model, 'lsn101.pth')
        #torch.save({'model': model.state_dict()}, 'lsn101_dict.pth')
        #best_acc = 100 * correct / total