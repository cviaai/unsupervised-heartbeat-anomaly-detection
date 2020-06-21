from tqdm import tqdm 
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import torch
import random


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTMCell(self.input_size, self.hidden_size)
        
        self.linear = nn.Linear(self.hidden_size, int(self.hidden_size/2))
        self.linear_1 = nn.Linear(int(self.hidden_size/2), self.output_size)
        self.dropout=nn.Dropout(0.5)
       
    def forward(self, input, future=0, y=None):
        outputs = []

        h_t = torch.zeros(input.size(0), self.hidden_size, dtype=torch.float32)
        c_t = torch.zeros(input.size(0), self.hidden_size, dtype=torch.float32)
        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            h_t, c_t = self.lstm(input_t, (h_t, c_t))
            output = self.linear_1(self.dropout(self.linear(h_t)))
            outputs += ([output])

        for i in range(future):
            if y is not None and random.random() > 0.5:
                output = y[:, [i]]  
            h_t, c_t = self.lstm(output, (h_t, c_t))
            output = self.linear_1(self.dropout(self.linear(h_t)))
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs