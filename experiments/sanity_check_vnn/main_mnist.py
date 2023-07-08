import os
import sys
import copy
import math
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from random import randrange
from math import sin, cos


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

import vn_layers2D as vn_layers

import torchvision
import torchvision.transforms as transforms


# set seed
seed = 10
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic=True

# cuda
cuda = 0
device = torch.device('cuda:'+ str(cuda))
froze_encoder = False

log_interval = 200

data_path = '/scratch/users/shizhehe/'

PC = False

batch_size = 64
lr = 0.1

class ToPC(object):

    def __call__(self, sample):
        ret = torch.tensor([[[i/sample.shape[1],j/sample.shape[2],sample[b][i][j]]for i in range(sample.shape[1])for j in range(sample.shape[2]) ]for b in range(sample.shape[0])])
        return ret
    

class get_model(nn.Module):

    def __init__(self, num_class=10, normal_channel=False):
        super(get_model, self).__init__()
        self.n_knn = 5 #args.n_knn
        
        self.conv1 = vn_layers.VNLinearLeakyReLU(1, 64//3)
        self.conv2 = vn_layers.VNLinearLeakyReLU(64//3, 64//3) 
        self.conv3 = vn_layers.VNLinearLeakyReLU(64//3, 128//3)
        self.conv4 = vn_layers.VNLinearLeakyReLU(128//3, 256//3)
        self.conv5 = vn_layers.VNLinearLeakyReLU(256//3+128//3+64//3+64//3, 1024//3, dim=4, share_nonlinearity=True)
        
        self.std_feature = vn_layers.VNStdFeature(1024//3*2, dim=3, normalize_frame=False)
        self.linear1 = nn.Linear((1024//3)*4, 512)
        
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, num_class)
        self.linear4 = nn.Linear(num_class, 3)
        

        self.pool1 = vn_layers.VNMaxPool(64//3)
        self.pool2 = vn_layers.VNMaxPool(64//3)
        self.pool3 = vn_layers.VNMaxPool(128//3)
        self.pool4 = vn_layers.VNMaxPool(256//3)


    def forward(self, x):
        batch_size = x.size(0)
        #x = x.unsqueeze(1)
        x = self.conv1(x)
        x1 = self.pool1(x)
        
        x = self.conv2(x)
        x2 = self.pool2(x)
        
        x = self.conv3(x)
        x3 = self.pool3(x)
        
        x = self.conv4(x)
        x4 = self.pool4(x)
        
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)
        
        num_points = x.size(-1)
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x, trans = self.std_feature(x)
        #swap x with trans?
        #x = x.view(batch_size, -1, num_points)
        
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)
        x = F.leaky_relu(self.bn1(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn2(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        feature_space = self.linear4(x)
        
        trans_feat = None
        return x, feature_space, trans_feat



class get_loss(torch.nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()


    def forward(self, pred, target, trans_feat, smoothing=True):
        ''' Calculate cross entropy loss, apply label smoothing if needed. '''

        target = target.contiguous().view(-1)

        if smoothing:
            eps = 0.2
            n_class = pred.size(1)

            one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(pred, target, reduction='mean')
            
        return loss


if PC:
    trainloader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(data_path, train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.RandomRotation((-90,90)),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,)),
                                ToPC(),
                                ])),
    batch_size=batch_size, shuffle=True)

    testloader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(data_path, train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,)),
                                ToPC(),
                                ])),
    batch_size=1, shuffle=True)

else:
    trainloader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(data_path, train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.RandomRotation((-90,90)),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,)),
                                ])),
    batch_size=batch_size, shuffle=True)

    testloader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(data_path, train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])),
    batch_size=1, shuffle=True)

print(f'Number of training examples: {len(trainloader.dataset)}')
print(f'Number of testing examples: {len(testloader.dataset)}')

dataiter = iter(trainloader)
images, labels = next(dataiter)
print("images", images.shape, "labels:", labels)

model = get_model().to(device)
criterion = get_loss()
optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=1e-4
        )
#optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4, amsgrad=True)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
global_epoch = 0
global_step = 0
best_instance_acc = 0.0
best_class_acc = 0.0
mean_correct = []

num_epochs = 4

print("Starting training!")

for epoch in range(num_epochs):  # loop over the dataset multiple times
    model.train()

    running_loss = 0.0
    pred_list = [] 
    label_list = []
    
    # train
    for i, data in enumerate(trainloader, 0):
    #for i, data in enumerate(tqdm(trainloader, total=len(trainloader), smoothing=0.9), 0):

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
       
        outputs, feature_space, feats = model(inputs)       
        loss = criterion(outputs, labels, feats)
        loss.backward()

        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % log_interval == 0:    # print every 200 mini-batches
            print('Epoch: {} - Loss: {:.6f}'.format(epoch + 1, running_loss / 2000))
            #print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            #print(outputs)
            running_loss = 0.0

    # test
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            outputs, feature_space, feats = model(data)
            
            test_loss += F.nll_loss(outputs, target, reduction='sum').item()  # sum up batch loss
            
            pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(testloader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))

    
    print('End of Epoch: {}'.format(epoch + 1))
    scheduler.step()


print('Finished Training')

torch.save(model.state_dict(), data_path + f'model.pt')

print('Saved model to disk')