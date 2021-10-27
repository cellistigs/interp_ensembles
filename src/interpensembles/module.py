import pytorch_lightning as pl
import torch
from pytorch_lightning.metrics import Accuracy

from .cifar10_models.densenet import densenet121, densenet161, densenet169
from .cifar10_models.googlenet import googlenet
from .cifar10_models.inception import inception_v3
from .cifar10_models.mobilenetv2 import mobilenet_v2
from .cifar10_models.resnet import resnet18, resnet34, resnet50
from .cifar10_models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from .schduler import WarmupCosineLR

all_classifiers = {
    "vgg11_bn": vgg11_bn,
    "vgg13_bn": vgg13_bn,
    "vgg16_bn": vgg16_bn,
    "vgg19_bn": vgg19_bn,
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "densenet121": densenet121,
    "densenet161": densenet161,
    "densenet169": densenet169,
    "mobilenet_v2": mobilenet_v2,
    "googlenet": googlenet,
    "inception_v3": inception_v3,
}


class CIFAR10Module(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy()

        self.model = all_classifiers[self.hparams.classifier]()

    def forward(self, batch):
        images, labels = batch
        predictions = self.model(images)
        loss = self.criterion(predictions, labels)
        accuracy = self.accuracy(predictions, labels)
        return loss, accuracy * 100

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
        scheduler = {
            "scheduler": WarmupCosineLR(
                optimizer, warmup_epochs=total_steps * 0.3, max_epochs=total_steps
            ),
            "interval": "step",
            "name": "learning_rate",
        }
        return [optimizer], [scheduler]

class CIFAR10EnsembleModule(CIFAR10Module):   
    """Customized module to train an ensemble of models independently.  

    """
    def __init__(self,nb_models,hparams):
        super().__init__(hparams)
        self.nb_models = nb_models
        self.hparams = hparams

        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy()

        self.models = torch.nn.ModuleList([all_classifiers[self.hparams.classifier]() for i in range(nb_models)]) ## now we add several different instances of the model. 
    
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
        loss = sum(losses)/self.nb_models ## calculate the average with pure python functions.    
        avg_accuracy = sum(accs)/self.nb_models

        self.log("loss/train", loss)
        self.log("acc/train", avg_accuracy*100)
        return loss


class CIFAR10InterEnsembleModule(CIFAR10Module):
    """Customized module to train a convex combination of a wide model and smaller models. 

    """
    def __init__(self,nb_models,lamb,hparams):
        self.nb_models = nb_models
        self.hparams = hparams

        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy()

        #self.interpmodel = # define this##  
     
    def forward(self,batch):
        """This forward function takes a convex combination of the original model and subnet models. 

        """
        images,labels = batch
        losses = []
        accs = []

        main_preds = self.interpmodel.base(images)
        main_loss = self.criterion(predictions,labels)
        losses.append(self.lamb*main_loss)
        accs.append(self.lamb*self.accuracy(main_preds,labels))

        for m in self.interpmodel.subnets:
            subnet_preds = m(predictions)
            subnet_loss = self.criterion(predictions,labels)
            losses.append((1-self.lamb)*(1/self.nb_models)*subnet_loss)
            accs.append((1-self.lamb)*(1/self.nb_models)*self.accuracy(subnet_preds,labels))
        loss = sum(losses)    
        avg_accuracy = sum(accs) 
        return loss, avg_accuracy*100

