## Test pytorch implementation of the CIFAR 10.1 dataset (v4 and v6). 
import pytest
import sys
import os 
import numpy as np
from torchvision import transforms as T
from PIL import Image

here = os.path.abspath(os.path.dirname(__file__))
#sys.path.insert(1,os.path.join(here,"../src/Pytorch_CIFAR10"))

from interpensembles.data import CIFAR10_1 
from torchvision.datasets import CIFAR10

class Test_CIFAR10_1:
    def test_init(self,tmpdir):
        downloads = tmpdir / "data"
        downloads.mkdir()
        CIFAR10_1(downloads)
        assert "data.npy" in [os.path.basename(l) for l in downloads.listdir()]
        assert "labels.npy" in [os.path.basename(l) for l in downloads.listdir()]

    def test_download(self,tmpdir):
        downloads = tmpdir / "data"
        downloads.mkdir()
        v4 = CIFAR10_1(downloads,version="v4")
        downloaded = [np.load(str(f)) for f in downloads.listdir()]
        for d in downloaded: 
            assert type(d) == np.ndarray
            assert d.shape[0] == 2021
            
        v6 = CIFAR10_1(downloads,version="v6")
        downloaded = [np.load(str(f)) for f in downloads.listdir()]
        for d in downloaded: 
            assert type(d) == np.ndarray
            assert d.shape[0] == 2000

    def test_parity(self,tmpdir):        
        """Check that data class has the fields of the same type as the original cifar dataset. 

        """
        downloads = tmpdir / "data"
        downloads.mkdir()
        cifar10 = CIFAR10(downloads,train = False,download = True) 
        cifar10_1 = CIFAR10_1(downloads)
        for attr,value in cifar10.__dict__.items():
            assert type(cifar10_1.__dict__[attr]) == type(value)
            if attr == "target":
                assert type(cifar10_1.__dict__[attr][0]) == type(value[0])
        
    def test_parity_transforms(self,tmpdir):    
        """Check that the field parity is maintained under transforms. 

        """
        downloads = tmpdir / "data"
        downloads.mkdir()
        transform = T.Compose(
            [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.5,0.5,0.5), (0.2,0.2,0.2)),
            ]
        )
        cifar10 = CIFAR10(downloads,train = False,download = True,transform = transform) 
        cifar10_1 = CIFAR10_1(downloads,transform = transform)
        for attr,value in cifar10.__dict__.items():
            assert type(cifar10_1.__dict__[attr]) == type(value)

    def test_classes(self,tmpdir):    
        """Save some images from both sets and make sure that the class labels make sense. 

        """
        downloads = tmpdir / "data"
        downloads.mkdir()
        cifar10 = CIFAR10(downloads,train = False,download = True,transform=None ) 
        cifar10_1 = CIFAR10_1(downloads,transform = None)
        class_indices = range(10) 
        for i in class_indices:
            cifar_ind = 0
            cifar_1_ind = 0
            cifar_classlabel = cifar10[cifar_ind][1]
            cifar_1_classlabel = cifar10_1[cifar_1_ind][1]

            while cifar_ind != i:
                cifar_ind+=1
            while cifar_1_ind != i:
                cifar_1_ind+=1
            cifarimage = cifar10[cifar_ind][0]
            cifar_1image = cifar10_1[cifar_1_ind][0]

            cifarimage.save(os.path.join(here,"test_mats","cifar10_ex_{}.jpg").format(cifar10[cifar_ind][1]))    
            cifar_1image.save(os.path.join(here,"test_mats","cifar10_1_ex_{}.jpg").format(cifar10_1[cifar_1_ind][1]))    



        

