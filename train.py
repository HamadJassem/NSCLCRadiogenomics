import pytorch_lightning as pl
from models.model import ImageClassifier
from models.tlmodel import TransferLearningModel
from utils.dataset import ChestCTDataset
from torch.utils.data import DataLoader
import pandas as pd
import os
from sklearn.model_selection import GroupShuffleSplit
from torchvision.transforms import ToTensor, Compose, Resize, Normalize
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import warnings
warnings.filterwarnings("ignore")

case_ids = []
labels = []
clinical_data = pd.read_csv('/home/hamad/pttut/clinical_data.csv')
data_dir = "/home/hamad/pttut/newpics"
data_dir2 = '/home/hamad/pttut/class1picslung'

case_ids = [name for name in os.listdir(data_dir)]
data = []
for image in case_ids:
    if not 'CT' in image:
        continue
    full_name = image.split('_CT')[0]
    if not full_name in clinical_data['Case ID'].values:
        continue
    label = clinical_data[clinical_data['Case ID'] == full_name].iloc[0,1]
    filepath = os.path.join(data_dir, image)
    data.append({'filepath': filepath, 'class_id': full_name, 'label': label})



df = pd.DataFrame(data)

gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
train_idxs, test_idxs = next(gss.split(df, groups=df['class_id'].values))
train_df = df.iloc[train_idxs]
test_df = df.iloc[test_idxs]

case_id_lung = []
for image in os.listdir(data_dir2):
    label = 1
    full_name = image.split('_')[0]
    filepath = os.path.join(data_dir2, image)
    case_id_lung.append({'filepath': filepath, 'class_id': full_name, 'label': label})

class_1 = pd.DataFrame(case_id_lung)
train_df = pd.concat([train_df, class_1], ignore_index=True)

train_idxs, valid_idxs = next(gss.split(train_df, groups=train_df['class_id'].values))
valid_df = train_df.iloc[valid_idxs]
train_df = train_df.iloc[train_idxs]

train_df = train_df.reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)




train_dataset = ChestCTDataset(train_df, transform=Compose([ToTensor(), Resize((224, 224))]))
train_loader = DataLoader(train_dataset, batch_size=32, num_workers=4)

valid_dataset = ChestCTDataset(valid_df, transform=Compose([ToTensor(), Resize((224, 224))]))
valid_loader = DataLoader(valid_dataset, batch_size=32, num_workers=4)

test_dataset = ChestCTDataset(test_df, transform=Compose([ToTensor(), Resize((224, 224))]))
test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4)

model = TransferLearningModel(num_classes=2, learning_rate=1e-3)

logger = TensorBoardLogger('tb_logs', name='my_model')

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='checkpoints',
    filename='best-checkpoint',
    save_top_k=3,
    mode='min',
)

early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=3,
    verbose=True,
    mode='min'
)

trainer = pl.Trainer(
    accelerator='gpu',
    max_epochs=10,
    logger=logger,
    callbacks=[checkpoint_callback,
               early_stopping_callback]
    )

trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

trainer.test(model, test_loader)

