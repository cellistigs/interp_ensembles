import os
import zipfile

from PIL import Image
import numpy as np
import pytorch_lightning as pl
import requests
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
import torchvision
from torchvision.datasets import CIFAR10
from tqdm import tqdm

def stream_download(dataurl,download_path):
    """helper function to monitor downloads. 
    :param dataurl: path where data is located. 
    :param download_path: local path (include filename) where we should write data. 

    """
    r = requests.get(dataurl,stream=True)
    total_size = int(r.headers.get("content-length", 0))
    block_size = 2 ** 20  # Mebibyte
    t = tqdm(total=total_size, unit="MiB", unit_scale=True)

    with open(download_path, "wb") as f:
        for data in r.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()

class CIFAR10_1(torchvision.datasets.vision.VisionDataset):
    """Pytorch wrapper around the CIFAR10.1 dataset (https://github.com/modestyachts/CIFAR-10.1)

    """
    def __init__(self,root_dir,version = "v6",transform = None,train = False,target_transform = None):
        """ 
        :param root_dir: path to store data files. 
        :param version: there is a v4 and v6 version of this data. 
        :param transform: an optional transform to be applied on PIL image samples.
        :param target_transform: an optional transform to be applied to targets. 
        **NB: there is an attribute "transforms" that we don't make use of in this class that the original CIFAR10 might.**
        """
        super().__init__(root_dir,transform = transform, target_transform = target_transform)
        #self.root = root_dir
        self.version = version
        #self.transform = transform
        self.train = train ## should always be false. 
        #self.target_transform = target_transform
        #self.transforms = None ## don't know what this parameter is.. 
        assert self.version in ["v4","v6"]
        ## Download data
        self.datapath, self.targetpath = self.download()
        ## Now get data and put it in memory: 
        self.data = np.load(self.datapath)
        self.targets = list(np.load(self.targetpath))
        ## Class-index mapping from original CIFAR10 dataset:  
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.class_to_idx = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}

    def download(self):    
        """Download data from github. 

        :returns: path where the data and label are located. 
        """
        root_path = "https://raw.github.com/modestyachts/CIFAR-10.1/master/datasets/"
        data_name = "cifar10.1_{}_data.npy".format(self.version)
        label_name = "cifar10.1_{}_labels.npy".format(self.version)

        dataurl = os.path.join(root_path,data_name)
        datadestination = os.path.join(self.root,"data.npy")
        stream_download(dataurl,datadestination)
        labelurl = os.path.join(root_path,label_name)
        labeldestination = os.path.join(self.root,"labels.npy")
        stream_download(labelurl,labeldestination)
        return datadestination,labeldestination

    def __len__(self):
        """Get dataset length:

        """
        return len(self.targets)

    def __getitem__(self,idx):
        """Get an item from the dataset: 

        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img = Image.fromarray(self.data[idx,:,:,:])    
        target = np.int64(self.targets[idx])
        if self.transform:
            img = self.transform(img)
        if self.target_transform:    
            target = self.target_transform(target)
        sample = (img,target)
        return sample


class CIFAR10_1Data(pl.LightningDataModule):
    def __init__(self,args,version = "v6"):
        super().__init__()
        self.hparams = args ## check these. 
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2471, 0.2435, 0.2616)
        self.version = version

    def train_dataloader(self):
        raise NotImplementedError

    def val_dataloader(self):
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )
        dataset = CIFAR10_1(root_dir=self.hparams.data_dir, train=False, transform=transform,version = self.version)
        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=False,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()

class CIFAR10Data(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.hparams = args
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2471, 0.2435, 0.2616)

    def download_weights():
        url = (
            "https://rutgers.box.com/shared/static/gkw08ecs797j2et1ksmbg1w5t3idf5r5.zip"
        )

        # Streaming, so we can iterate over the response.
        r = requests.get(url, stream=True)

        # Total size in Mebibyte
        total_size = int(r.headers.get("content-length", 0))
        block_size = 2 ** 20  # Mebibyte
        t = tqdm(total=total_size, unit="MiB", unit_scale=True)

        with open("state_dicts.zip", "wb") as f:
            for data in r.iter_content(block_size):
                t.update(len(data))
                f.write(data)
        t.close()

        if total_size != 0 and t.n != total_size:
            raise Exception("Error, something went wrong")

        print("Download successful. Unzipping file...")
        path_to_zip_file = os.path.join(os.getcwd(), "state_dicts.zip")
        directory_to_extract_to = os.path.join(os.getcwd(), "cifar10_models")
        with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
            zip_ref.extractall(directory_to_extract_to)
            print("Unzip file successful!")

    def train_dataloader(self):
        transform = T.Compose(
            [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )
        dataset = CIFAR10(root=self.hparams.data_dir, train=True, transform=transform,download = True)
        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )
        dataset = CIFAR10(root=self.hparams.data_dir, train=False, transform=transform,download = True)
        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=False,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()
