import pytorch_lightning as pl
import torch
try:
    from pytorch_lightning.metrics import Accuracy
except ModuleNotFoundError:    
    from torchmetrics import Accuracy

from .cifar10_models.densenet import densenet121, densenet161, densenet169
from .cifar10_models.googlenet import googlenet
from .cifar10_models.inception import inception_v3
from .cifar10_models.mobilenetv2 import mobilenet_v2
from .cifar10_models.resnet import resnet10, resnet18, resnet26, resnet34, resnet42, resnet52, resnet50, wideresnet18, wideresnet18_4,widesubresnet18,wideresnet18_4_grouplinear,narrowresnet10_32,narrowresnet10_16,narrowresnet10_8,narrowresnet10_4,narrowresnet10_2,resnet10,narrowresnet12_32,narrowresnet12_16,narrowresnet12_8,narrowresnet12_4,narrowresnet12_2,resnet12,narrowresnet14_32,narrowresnet14_16,narrowresnet14_8,narrowresnet14_4,narrowresnet14_2,resnet14,narrowresnet16_32,narrowresnet16_16,narrowresnet16_8,narrowresnet16_4,narrowresnet16_2,resnet16,narrowresnet18_32,narrowresnet18_16,narrowresnet18_8,narrowresnet18_4,narrowresnet18_2,resnet18
from .cifar10_models.resnet_cifar import resnet8_cf,resnet14_cf,resnet20_cf,resnet26_cf,resnet32_cf,resnet38_cf,resnet44_cf 
from .cifar10_models.wideresnet_28 import wideresnet28_10
from .cifar10_models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from .cifar10_models.efficientnet import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3
from .cifar10_models.pyramidnet import pyramidnet272,pyramidnet164,pyramidnet110
from .cifar10_models.shake_shake import shake_resnet26_2x96d
from .cifar10_models.resnexst import resnexst50_4x16d_cifar
from .cifar10_models.senet import se_resnet164
from .schduler import WarmupCosineLR

all_classifiers = {
    "vgg11_bn": vgg11_bn,
    "vgg13_bn": vgg13_bn,
    "vgg16_bn": vgg16_bn,
    "vgg19_bn": vgg19_bn,
    "wideresnet18":wideresnet18,
    "wideresnet18_4":wideresnet18_4,
    "wideresnet18_4_grouplinear":wideresnet18_4_grouplinear,
    "wideresnet28_10":wideresnet28_10,
    "resnet8_cifar": resnet8_cf,
    "resnet14_cifar": resnet14_cf,
    "resnet20_cifar": resnet20_cf,
    "resnet26_cifar": resnet26_cf,
    "resnet32_cifar": resnet32_cf,
    "resnet44_cifar": resnet44_cf,
    "narrowresnet10_32":narrowresnet10_32,
    "narrowresnet10_16":narrowresnet10_16,
    "narrowresnet10_8":narrowresnet10_8,
    "narrowresnet10_4":narrowresnet10_4,
    "narrowresnet10_2":narrowresnet10_2,
    "resnet10":resnet10,
    "narrowresnet12_32":narrowresnet12_32,
    "narrowresnet12_16":narrowresnet12_16,
    "narrowresnet12_8":narrowresnet12_8,
    "narrowresnet12_4":narrowresnet12_4,
    "narrowresnet12_2":narrowresnet12_2,
    "resnet12":resnet12,
    "narrowresnet14_32":narrowresnet14_32,
    "narrowresnet14_16":narrowresnet14_16,
    "narrowresnet14_8":narrowresnet14_8,
    "narrowresnet14_4":narrowresnet14_4,
    "narrowresnet14_2":narrowresnet14_2,
    "resnet14":resnet14,
    "narrowresnet16_32":narrowresnet16_32,
    "narrowresnet16_16":narrowresnet16_16,
    "narrowresnet16_8":narrowresnet16_8,
    "narrowresnet16_4":narrowresnet16_4,
    "narrowresnet16_2":narrowresnet16_2,
    "resnet16":resnet16,
    "narrowresnet18_32":narrowresnet18_32,
    "narrowresnet18_16":narrowresnet18_16,
    "narrowresnet18_8":narrowresnet18_8,
    "narrowresnet18_4":narrowresnet18_4,
    "narrowresnet18_2":narrowresnet18_2,
    "resnet18": resnet18,
    "resnet26": resnet26,
    "resnet34": resnet34,
    "resnet42": resnet42,
    "resnet50": resnet50,
    "resnet52": resnet52,
    "densenet121": densenet121,
    "densenet161": densenet161,
    "densenet169": densenet169,
    "mobilenet_v2": mobilenet_v2,
    "googlenet": googlenet,
    "inception_v3": inception_v3,
    "efficientnet_b0": efficientnet_b0,
    "efficientnet_b1": efficientnet_b1,
    "efficientnet_b2": efficientnet_b2,
    "efficientnet_b3": efficientnet_b3,
    "pyramidnet_272": pyramidnet272,
    "pyramidnet_164": pyramidnet164,
    "pyramidnet_110": pyramidnet110,
    "shake_resnet26_2x96d": shake_resnet26_2x96d,
    "se_resnet164": se_resnet164,
    "resnexst50_4x16d_cifar": resnexst50_4x16d_cifar 
}


class CIFAR10_Models(pl.LightningModule):
    """Abstract base class for CIFAR10 Models

    """
    def __init__(self,hparams):
        super().__init__()
        self.hparams.update(hparams)
    def forward(x):    
        raise NotImplementedError
    def training_step():
        raise NotImplementedError

    def calibration():
        """Calculates binned calibration metrics given 

        """

    def validation_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("loss/val", loss)
        self.log("acc/val", accuracy)

    def test_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("acc/test", accuracy)

    def setup_scheduler(self,optimizer,total_steps):
        """Chooses between the cosine learning rate scheduler that came with the repo, or step scheduler based on wideresnet training. 

        """
        if self.hparams.scheduler in [None,"cosine"]: 
            scheduler = {
                "scheduler": WarmupCosineLR(
                    optimizer, warmup_epochs=self.hparams.warmup_epochs, max_epochs=total_steps
                ),
                "interval": "step",
                "name": "learning_rate",
            }
        elif self.hparams.scheduler == "step":    
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones = [60,120,160], gamma = 0.1, last_epoch=-1
                ),
                "interval": "epoch",
                "frequency":1,
                "name": "learning_rate",
                }
        return scheduler    

class CIFAR10LinearGroupModule(CIFAR10_Models):
    """Replaces the final layer with a LogSoftmaxGroupLinear layer, and correspondingly changes the loss. . 

    """
    def __init__(self, hparams):
        super().__init__(hparams)

        self.criterion = torch.nn.NLLLoss()
        self.accuracy = Accuracy()
        assert self.hparams.classifier.endswith("grouplinear"), "will give strange results for those that don't have softmax grouplinear output"
        self.model = all_classifiers[self.hparams.classifier]()

    def forward(self, batch):
        images, labels = batch
        predictions = self.model(images)
        loss = self.criterion(predictions, labels)
        accuracy = self.accuracy(predictions, labels)
        return loss, accuracy * 100

    def calibration(self, batch):
        images, labels = batch
        predictions = self.model(images)
        return predictions, labels

    def training_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("loss/train", loss)
        self.log("acc/train", accuracy)
        return loss

    def validation_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("loss/val", loss)
        self.log("acc/val", accuracy)

    def test_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("acc/test", accuracy)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            momentum=0.9,
            nesterov=True,
        )
        total_steps = self.hparams.max_epochs * len(self.train_dataloader())
        scheduler = self.setup_scheduler(optimizer,total_steps)
        return [optimizer], [scheduler]

class CIFAR10Module(CIFAR10_Models):
    def __init__(self, hparams):
        super().__init__(hparams)
        print(hparams)
        print(self.hparams)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy()

        self.model = all_classifiers[self.hparams.classifier]()

    def forward(self, batch):
        images, labels = batch
        predictions = self.model(images)
        loss = self.criterion(predictions, labels)
        accuracy = self.accuracy(predictions, labels)
        return loss, accuracy * 100

    def calibration(self,batch,use_softmax = True):
        """Like forward, but just exit with the softmax predictions and labels. . 
        """
        softmax = torch.nn.Softmax(dim = 1)
        images, labels = batch
        predictions = self.model(images)
        if use_softmax:
            smpredictions = softmax(predictions)
        else:    
            smpredictions = predictions
        return smpredictions,labels

    def training_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("loss/train", loss)
        self.log("acc/train", accuracy)
        return loss

    def validation_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("loss/val", loss)
        self.log("acc/val", accuracy)

    def test_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("acc/test", accuracy)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            momentum=0.9,
            nesterov=True,
        )
        total_steps = self.hparams.max_epochs * len(self.train_dataloader())
        scheduler = self.setup_scheduler(optimizer,total_steps)
        return [optimizer], [scheduler]

class CIFAR10EnsembleModule(CIFAR10_Models):   
    """Customized module to train an ensemble of models independently.  

    """
    def __init__(self,nb_models,hparams):
        super().__init__(hparams)
        self.nb_models = nb_models

        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy()

        self.models = torch.nn.ModuleList([all_classifiers[self.hparams.classifier]() for i in range(nb_models)]) ## now we add several different instances of the model. 
        #del self.model
    
    def forward(self,batch):
        """for forward, we want to take the softmax, aggregate the ensemble output, and then take the logit.  

        """
        images, labels = batch
        softmax = torch.nn.Softmax(dim = 1)

        losses = []
        accs = []
        softmaxes = []
        for m in self.models:
            predictions = m(images)
            normed = softmax(predictions)
            softmaxes.append(normed)
        gmean = torch.exp(torch.mean(torch.log(torch.stack(softmaxes)),dim = 0)) ## implementation from https://stackoverflow.com/questions/59722983/how-to-calculate-geometric-mean-in-a-differentiable-way   
        ## we can pass this  through directly to the accuracy function. 
        tloss = self.criterion(gmean,labels)## beware: this is a transformed input, don't evaluate on test loss of ensembles. 
        accuracy = self.accuracy(gmean,labels)
        return tloss,accuracy*100

    def calibration(self,batch):
        """Like forward, but just exit with the predictions and labels. . 
        """
        images, labels = batch
        softmax = torch.nn.Softmax(dim = 1)

        losses = []
        accs = []
        softmaxes = []
        for m in self.models:
            predictions = m(images)
            normed = softmax(predictions)
            softmaxes.append(normed)
        gmean = torch.exp(torch.mean(torch.log(torch.stack(softmaxes)),dim = 0)) ## implementation from https://stackoverflow.com/questions/59722983/how-to-calculate-geometric-mean-in-a-differentiable-way   
        return gmean,labels

    def training_step(self, batch, batch_nb):
        """When we train, we want to train independently. 
        """
        
        images, labels = batch
        losses = []
        accs = []
        for m in self.models:
            predictions = m(images) ## this just a bunch of unnormalized scores? 
            mloss = self.criterion(predictions, labels)
            accuracy = self.accuracy(predictions,labels)
            losses.append(mloss)
            accs.append(accuracy) 
        loss = sum(losses) ## calculate the sum with pure python functions.    
        avg_accuracy = sum(accs)/self.nb_models

        self.log("loss/train", loss)
        self.log("acc/train", avg_accuracy*100)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.models.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            momentum=0.9,
            nesterov=True,
        )
        total_steps = self.hparams.max_epochs * len(self.train_dataloader())
        scheduler = self.setup_scheduler(optimizer,total_steps)
        return [optimizer], [scheduler]

class CIFAR10InterEnsembleModule(CIFAR10_Models):
    """Customized module to train a convex combination of a wide model and smaller models. 

    """
    def __init__(self,lamb,hparams):

        super().__init__(hparams)
        self.lamb = lamb
        self.hparams = hparams

        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy()

        #self.interpmodel = # define this##  
        self.model = all_classifiers[self.hparams.classifier]()
        self.basemodel = wideresnet18()
        self.submodels = torch.nn.ModuleList([widesubresnet18(self.basemodel,i) for i in range(4)])
     
    def forward(self,batch): 
        """First pass forward: try treating like a standard ensemble? 

        """
        images, labels = batch
        softmax = torch.nn.Softmax(dim = 1)

        losses = []
        accs = []
        softmaxes = []
        for m in self.submodels:
            predictions = m(images)
            normed = softmax(predictions)
            softmaxes.append(normed)
        ## todo: should we add the main model predictions too?

        gmean = torch.exp(torch.mean(torch.log(torch.stack(softmaxes)),dim = 0)) ## implementation from https://stackoverflow.com/questions/59722983/how-to-calculate-geometric-mean-in-a-differentiable-way   
        bigpred = self.basemodel(images)
        bignormed = softmax(bigpred)
        grand_mean = torch.sum(torch.stack([self.lamb*bignormed,(1-self.lamb)*gmean]),dim = 0)
        ## we can pass this  through directly to the accuracy function. 
        tloss = self.criterion(grand_mean,labels)## beware: this is a transformed input, don't evaluate on test loss of ensembles. 
        accuracy = self.accuracy(grand_mean,labels)
        return tloss,accuracy*100

    def calibration(self,batch):
        """Like forward, but just exit with the predictions and labels. . 
        """
        images, labels = batch
        softmax = torch.nn.Softmax(dim = 1)

        losses = []
        accs = []
        softmaxes = []
        for m in self.submodels:
            predictions = m(images)
            normed = softmax(predictions)
            softmaxes.append(normed)
        ## todo: should we add the main model predictions too?

        gmean = torch.exp(torch.mean(torch.log(torch.stack(softmaxes)),dim = 0)) ## implementation from https://stackoverflow.com/questions/59722983/how-to-calculate-geometric-mean-in-a-differentiable-way   
        bigpred = self.basemodel(images)
        bignormed = softmax(bigpred)
        grand_mean = torch.sum(torch.stack([self.lamb*bignormed,(1-self.lamb)*gmean]),dim = 0)
        return grand_mean,labels

    def training_step(self,batch,batch_nb):
        """This training_step function takes a convex combination of the original model and subnet models. 

        """
        images,labels = batch
        losses = []
        accs = []

        main_preds = self.basemodel(images)
        main_loss = self.criterion(main_preds,labels)
        losses.append(self.lamb*main_loss)
        accs.append(self.lamb*self.accuracy(main_preds,labels))
        nb_subnets = self.submodels[0].nb_subnets

        for m in self.submodels:
            subnet_preds = m(images)
            subnet_loss = self.criterion(subnet_preds,labels)
            losses.append((1-self.lamb)*(1/nb_subnets)*subnet_loss)
            accs.append((1-self.lamb)*(1/nb_subnets)*self.accuracy(subnet_preds,labels))
        loss = sum(losses)    
        avg_accuracy = sum(accs) 
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            list(self.submodels.parameters())+list(self.basemodel.parameters()),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            momentum=0.9,
            nesterov=True,
        )
        total_steps = self.hparams.max_epochs * len(self.train_dataloader())
        scheduler = self.setup_scheduler(optimizer,total_steps)
        return [optimizer], [scheduler]

