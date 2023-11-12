import torch
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor
# create a pytorch dataset class

class ChestCTDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform if transform is not None else ToTensor()
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        img_path = self.dataframe.iloc[index, 0]
        label = self.dataframe.iloc[index, 2]
        
        try:
            image = Image.open(img_path).convert('L') 
            image = np.stack((image,)*3, axis=-1) 
        except IOError:
            print('Error reading image: {}'.format(img_path))
            return None       

        image = self.transform(image)    
        label = torch.tensor(label, dtype=torch.long)
        one_hot_label = torch.zeros(2)
        one_hot_label[label] = 1
        sample = {'image': image, 'label': one_hot_label}
        return sample
    
    
        