import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

#-------------------------------------------------------------------------------- 

class NIHDataset(Dataset):
    
    #-------------------------------------------------------------------------------- 
    def __init__(self, root, split="train", is_transform=False, img_size=(512, 512), augmentations=None):
    
        self.listImagePaths = []
        self.listImageLabels = []
        self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(size=img_size),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

        if split == "train":
            pathImageDirectory = os.path.join(root,"training_images/")
            pathDatasetFile = os.path.join(root,"nih_train.txt")

        if split == "test" or split == "val":
            pathImageDirectory = os.path.join(root,"nih_test/")
            pathDatasetFile = os.path.join(root,"nih_test.txt")
    
        #---- Open file, get image paths and labels
    
        fileDescriptor = open(pathDatasetFile, "r")
        
        #---- get into the loop
        line = True
        
        while line:
                
            line = fileDescriptor.readline()
            
            #--- if not empty
            if line:
          
                lineItems = line.split()
                
                imagePath = os.path.join(pathImageDirectory, lineItems[0])
                imageLabel = lineItems[1:]
                imageLabel = [int(i) for i in imageLabel]
                if sum(imageLabel) > 0:
                    imageLabel.append(0)
                else:
                    imageLabel.append(1)    
                
                self.listImagePaths.append(imagePath)
                self.listImageLabels.append(imageLabel)  
            
        fileDescriptor.close()
    
    #-------------------------------------------------------------------------------- 
    
    def __getitem__(self, index):
        
        imagePath = self.listImagePaths[index]
        
        imageData = Image.open(imagePath).convert('RGB')
        imageLabel= torch.FloatTensor(self.listImageLabels[index])  
        
        if self.transform != None: 
            imageData = self.transform(imageData)
        return imageData, imageLabel
        
    #-------------------------------------------------------------------------------- 
    
    def __len__(self):
        
        return len(self.listImagePaths)
    
 #-------------------------------------------------------------------------------- 
    