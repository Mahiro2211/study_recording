#!/usr/bin/env python
# coding: utf-8

# 经卷积之后的矩阵尺寸大小
# $N = （W - F + 2P）/S + 1$
# 
# $W * W$
# 
# $filter$
# 
# $Stride$
# 
# $padding$
# * 池化不会影响深度只会影响高度和宽度
# * tensor的通道 $[batch , channel , height , weight]$

# In[65]:


import torch.nn as nn
import torch.nn.functional as F
import pdb
import os
from PIL import Image


# In[2]:


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet , self).__init__()
        self.conv1 = nn.Conv2d(3 , 16 , 5)
        #pdb.set_trace() # 设置断点
        self.pool1 = nn.MaxPool2d(2 , 2)
        self.conv2 = nn.Conv2d(16 , 32 , 5)
        self.pool2 = nn.MaxPool2d(2 , 2)
        self.fc1 = nn.Linear(32 * 5 * 5 , 120)
        self.fc2 = nn.Linear(120 , 84)
        self.fc3 = nn.Linear(84 , 10)
    def forward(self , x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view( -1 , 32 * 5 * 5 )
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# In[3]:


import torch


# In[4]:


input1 = torch.rand([32 , 3 , 32 , 32])
model = LeNet()
print(model)
output = model(input1)


# In[7]:


import torch.optim as optim
import torchvision.transforms as  transforms
import matplotlib.pyplot as plt
import numpy as np
import torchvision


# In[113]:


transform = transforms.Compose(
[
    transforms.Resize((32 , 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5 ,0.5) , (0.5 ,0.5 , 0.5))
])

    train_set = torchvision
# In[114]:


train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=36,
                                               shuffle=True, num_workers=4)


# In[115]:


val_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=False, transform=transform)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=5000,
                                             shuffle=False, num_workers=4)
val_data_iter = iter(val_loader)
val_image, val_label = next(val_data_iter)


# In[116]:


classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
def imshow(img):
    img = img / 2 + 0.5     # 反标准化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# In[117]:


net = LeNet()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters() , lr = 1e-3)


# In[118]:


for epoch in range(5):  # loop over the dataset multiple times

       running_loss = 0.0
       for step, data in enumerate(train_loader, start=0):
           # get the inputs; data is a list of [inputs, labels]
           inputs, labels = data

           # zero the parameter gradients
           optimizer.zero_grad()
           # forward + backward + optimize
           outputs = net(inputs)
           loss = loss_function(outputs, labels)
           loss.backward()
           optimizer.step()

           # print statistics
           running_loss += loss.item()
           if step % 500 == 499:    # print every 500 mini-batches
               with torch.no_grad():
                   outputs = net(val_image)  # [batch, 10]
                   predict_y = torch.max(outputs, dim=1)[1] # index
                   accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)

                   print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                         (epoch + 1, step + 1, running_loss / 500, accuracy))
                   running_loss = 0.0


# In[119]:


save_path = "./Lenet.pth"
torch.save(net.state_dict() , save_path)


# In[120]:


im = Image.open('1.jpeg')


# In[121]:


im = transform(im)


# In[122]:


type(im)


# In[123]:


im = torch.unsqueeze(im , dim = 0)
im.size()


# In[125]:


with torch.no_grad():
    outputs = model(im)
    predict = torch.softmax(outputs , dim=1)
print(predict)


# In[ ]:




