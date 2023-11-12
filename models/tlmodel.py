import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics import Accuracy
from torchvision import models

class TransferLearningModel(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=1e-3):
        super(TransferLearningModel, self).__init__()
        # Model architecture
        self.model = models.resnet18(pretrained=True)
        
        for param in self.model.parameters():
            param.requires_grad = False
            
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        
        # Metrics
        self.accuracy = Accuracy(num_classes=num_classes, task='multiclass')
        self.valid_acc = Accuracy(num_classes=num_classes, task='multiclass')
        self.learning_rate = learning_rate

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x = batch['image']
        y = batch['label']
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss)        
        self.accuracy(logits, y)
        self.log('train_acc', self.accuracy, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['image']
        y = batch['label']
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('val_loss', loss, prog_bar=True)
        self.valid_acc(logits, y)
        self.log('val_acc', self.valid_acc, prog_bar=True, on_step=True, on_epoch=True)
        
    def test_step(self, batch, batch_idx):
        x = batch['image']
        y = batch['label']
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('test_loss', loss, prog_bar=True)
        self.valid_acc(logits, y)
        self.log('test_acc', self.valid_acc, prog_bar=True, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
