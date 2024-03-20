import torch
from torch import nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class block(nn.Module):
  def __init__(self,filters,kernel_size=3,subsample=False):
    super().__init__()
    # if s equals 1, we don't subsample and we use all the input channels. if it is 0.5, we subsample in the first conv layer
    s = 0.5 if subsample else 1.0
    self.conv1 = nn.Conv2d(int(filters*s), filters, kernel_size=kernel_size, stride=int(1/s), padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(filters,track_running_stats=True)
    self.relu1 = nn.ReLU()
    self.conv2 = nn.Conv2d(filters, filters, kernel_size=kernel_size, stride=1, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(filters,track_running_stats=True)
    self.relu2 = nn.ReLU()
    self.downsample = nn.AvgPool2d(kernel_size=1,stride=2)
    # Initialise weights according to the method described in 
    # “Delving deep into rectifiers: Surpassing human-level performance on ImageNet 
    # classification” - He, K. et al. (2015)
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)   
  # x is the input tesnsor and z is the output before going through the non-linearity (activation function, in this case the Relu function)
  def shortcut(self,z,x):
    if (x.shape == z.shape):
      return x+z
    d = self.downsample(x)
    p = torch.mul(d,0)
    return z + torch.cat((d, p), dim=1)
  def forward(self,x,shortcuts=False):
    z = self.conv1(x)
    z = self.bn1(z)
    z = self.relu1(z)
    z = self.conv2(z)
    z = self.bn2(z)
    if shortcuts:
      z = self.shortcut(z,x)
    z = self.relu2(z)
    return z

class ResNet(nn.Module):
  def __init__(self,n,shortcuts=False):
    super().__init__()
    self.shortcuts = shortcuts
    # Input
    self.convIn = nn.Conv2d(3,16,kernel_size=3,stride=1,padding=1,bias=False)
    self.bnIn = nn.BatchNorm2d(16,track_running_stats=True)
    self.reulUn = nn.ReLU()

    # Stack 1
    self.stack1 = nn.ModuleList([block(16,subsample=False) for _ in range (n)])

    # Stack 2
    self.stack2a = block(32,subsample=True)
    self.stack2b = nn.ModuleList([block(32,subsample=False) for _ in range (n-1)])

    # Stack 3
    self.stack3a = block(64,subsample=True)
    self.stack3b = nn.ModuleList([block(64,subsample=False) for _ in range (n-1)])

    # Others
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fcOut   = nn.Linear(64, 10, bias=True)
    self.softmax = nn.LogSoftmax(dim=1)
    

    for m in self.modules():
      if isinstance(m, nn.Linear):
        nn.init.kaiming_normal(m.weight)
        m.bias.data.zero_()     
  def forward(self,x):
    z = self.convIn(x)
    z = self.bnIn(z)
    z = self.reulUn(z)
    for layer in self.stack1:
      z = layer(z,shortcuts=self.shortcuts)
    z = self.stack2a(z,shortcuts=self.shortcuts)
    for layer in self.stack2b:
      z = layer(z,shortcuts=self.shortcuts)
    z = self.stack3a(z,shortcuts=self.shortcuts)
    for layer in self.stack3b:
      z = layer(z,shortcuts=self.shortcuts)
    z = self.avgpool(z)
    z = z.view(z.size(0),-1)
    z = self.fcOut(z)
    z = self.softmax(z)
    return z