import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

def get_data_loaders(data_dir,batch_size,train_transform,test_transform,shuffle=True,num_workers=2,pin_memory=False):
  train_dataset = datasets.CIFAR10(data_dir,train=True,download=True,transform=train_transform)
  test_dataset = datasets.CIFAR10(data_dir,train=False,download=True,transform=test_transform) 
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers, pin_memory=pin_memory)
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers, pin_memory=pin_memory)
  return (train_loader,test_loader)