## Create RFF models. 
import torch
from torch import nn
import numpy as np

class RFF_Projection(nn.Module):
    def __init__(self,input_size,project_size):
        """Create an rff projection

        """
        super().__init__()
        self.weights = nn.init.normal_(torch.Tensor(project_size,input_size),1/np.sqrt(self.sigma*input_size))
        self.offset = nn.init.uniform_(torch.Tensor(size_out),0,2*np.pi)
        self.weights.requires_grad = False
        self.offset.requires_grad = False
        
    def forward(self,x):    
        return torch.cos(torch.mm(x,self.weights.t())+self.offset)

class RFF(nn.Module): 
    """Random Fourier Features module.

    """
    def __init__(self,input_size,project_size,output_size,sigma = 1):
        self.project = RFF_Projection(input_size,project_size) 
        self.regress = nn.Linear(project_size,output_size) 

    def forward(self,x):    
        proj = self.project(x)
        out = self.regress(proj)
        return out
