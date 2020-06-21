import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
#import ot
from abc import ABC, abstractmethod

dimZ = 200 # Considering face reconstruction task, which size of representation seems reasonable?
batch_size=1

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.0)
        
        self.encoder=nn.Sequential()
        self.encoder.add_module('linear1',nn.Linear(1*2236*3,500))
        self.encoder.add_module('relu_1',nn.ReLU())
        
#         self.encoder.add_module('linear11',nn.Linear(1000,500))
#         self.encoder.add_module('relu_11',nn.ReLU())
        
#         self.encoder.add_module('linear12',nn.Linear(2000,1000))
#         self.encoder.add_module('relu_12',nn.ReLU())
        
        self.encoder.add_module('linear2',nn.Linear(500,100))
        self.encoder.add_module('active1',nn.Sigmoid())
        self.encoder.apply(init_weights)
    def forward(self, x):

      latent_code =self.encoder(x)


      return latent_code