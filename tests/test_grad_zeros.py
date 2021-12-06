## Test if we can use pytorch/ptl to manually zero out weights during training. 
import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F



class Single_Layer(pl.LightningModule):
    """Builds a single layer. 

    """
    def __init__(self):
        in_channels = 16
        out_channels = 16
        kernel_size = 3
        super().__init()__
        self.layer = torch.nn.Conv2d(in_channels,out_channels,kernel_size)
    def forward(self,x):    
        return self.layer(x)
    def training_step(self,batch,batch_idx):
        
        
