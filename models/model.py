import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics import Accuracy

class ImageClassifier(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=1e-3):
        super(ImageClassifier, self).__init__()
        # Model architecture
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        dummy_input = torch.zeros(1, 3, 224, 224)
        dummy_input = self.pool(F.relu(self.conv1(dummy_input)))
        dummy_input = self.pool(F.relu(self.conv2(dummy_input)))
        flattened_size = torch.flatten(dummy_input, 1).shape[1]
        
        self.fc1 = nn.Linear(flattened_size, 512)  # Adjust the input features to match your image size
        self.fc2 = nn.Linear(512, num_classes)

        # Metrics
        self.accuracy = Accuracy(num_classes=num_classes, task='multiclass')
        self.valid_acc = Accuracy(num_classes=num_classes, task='multiclass')
        self.learning_rate = learning_rate

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # Flatten the tensor for the fully connected layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
