import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

#-------------------------------------------------------------------------------- 
# nih_classes = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
#                 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 
#                 'Hernia', 'Normal', 'Covid']
#nih_classes = [ 'Ate', 'Car', 'Eff', 'Inf', 'Mas', 'Nod', 'Pne',
#                 'Pnt', 'Con', 'Ede', 'Emp', 'Fib', 'Ple', 
#                 'Her', 'Nor', 'Cov']
# Ate_Car_Eff_Inf_Mas_Nod_Pne_Pnt_Con_Ede_Emp_Fib_Ple_Her_Nor

class CovidNIHDatasetPartial(Dataset):
    
    #-------------------------------------------------------------------------------- 
    def __init__(self, root, split="train", is_transform=False, img_size=(512, 512), nih_labels=[], 
            augmentations=None, whatsapp_data=False, nih_normal = False,
            nih_pne = False):
    
        self.listImagePaths = []
        self.listImageLabels = []
        self.listWAImagePaths = []
        self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(size=img_size),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

        self.nih_normal_flag = nih_normal
        self.nih_pne_flag = nih_pne

        self.nih_labels = self.convert_label_toints(nih_labels)
        self.whatsapp_data = whatsapp_data

        if split == "train":
            pathImageDirectory = os.path.join(root,"training_images/")
            pathDatasetFile = os.path.join(root,"nih_train.txt")

        if split == "test" or split == "val":
            pathImageDirectory = os.path.join(root,"nih_test/")
            pathDatasetFile = os.path.join(root,"nih_test.txt")
            if self.whatsapp_data == True:
                pathWAImgDir = os.path.join(root,"whatsapp_nih_test/")
    
        #---- Open file, get image paths and labels
    
        fileDescriptor = open(pathDatasetFile, "r")
        
        self.load_covid_images(split)

        nih_cnt = 0
        #---- get into the loop
        line = True
        
        while line:
                
            line = fileDescriptor.readline()
            
            #--- if not empty
            if line:
          
                lineItems = line.split()
                
                imagePath = os.path.join(pathImageDirectory, lineItems[0])
                imageLabel = lineItems[1:]

                final_image_label = [0] * len(self.nih_labels)

                if self.whatsapp_data == True:
                    imagePathWhatsapp = os.path.join(pathWAImgDir, lineItems[0])
                    imagePathWhatsapp = imagePathWhatsapp.replace(".png", ".jpg")

                for i in range(len(self.nih_labels)):
                    final_image_label[i] = int(imageLabel[self.nih_labels[i]])                        

                imageLabel = [int(i) for i in imageLabel]
                if sum(final_image_label) > 0:
                    self.listImagePaths.append(imagePath)
                    self.listImageLabels.append(imageLabel)  
                    if self.whatsapp_data == True:
                        self.listWAImagePaths.append(imagePathWhatsapp)
                    nih_cnt += 1    
            
        fileDescriptor.close()
        print("NIH data added : {}".format(nih_cnt))
    
    #-------------------------------------------------------------------------------- 
    
    def __getitem__(self, index):
        
        imagePath = self.listImagePaths[index]
        
        imageData = Image.open(imagePath).convert('RGB')
        imageLabel= torch.FloatTensor(self.listImageLabels[index])  
        
        if self.transform != None: 
            imageData = self.transform(imageData)

        if self.whatsapp_data == True:
            imagePathWA = self.listWAImagePaths[index]
            imageDataWA = Image.open(imagePathWA).convert('RGB')
            if self.transform != None: 
                imageDataWA = self.transform(imageDataWA)
            return imageData, imageDataWA, imageLabel            

        return imageData, imageLabel
        
    #-------------------------------------------------------------------------------- 
    
    def __len__(self):
        
        return len(self.listImagePaths)

    #-------------------------------------------------------------------------------- 
    # 
    def convert_label_toints(self, labels):
        nih_classes = [ 'Ate', 'Car', 'Eff', 'Inf', 'Mas', 'Nod', 'Pne',
                        'Pnt', 'Con', 'Ede', 'Emp', 'Fib', 'Ple', 
                        'Her', 'Nor', 'Cov']                
        int_labels = []                        
        for x in labels:
            if x != 'Nor' and x != 'Pne':
                int_labels.append(nih_classes.index(x))
            elif x == 'Nor':
                if self.nih_normal_flag != False: 
                    int_labels.append(nih_classes.index(x))
            else:                    
                if self.nih_pne_flag != False:
                    int_labels.append(nih_classes.index(x))
        return int_labels    
    
 #-------------------------------------------------------------------------------- 
    

    def load_covid_images(self, split):
        covid_pathDirData = "/scratch/mariamma/xraysetu/dataset/covidx3_upsampled/"
        if split == "train":
            pathDatasetFile = "/scratch/mariamma/xraysetu/dataset/covid_train.txt"
            pathImageDirectory = os.path.join(covid_pathDirData,"train") 
        elif split == "val":
            pathDatasetFile = "/scratch/mariamma/xraysetu/dataset/covid_val.txt"
            pathImageDirectory = os.path.join(covid_pathDirData,"test")
        elif split == "test":    
            pathDatasetFile = "/scratch/mariamma/xraysetu/dataset/covid_test.txt"
            pathImageDirectory = "/scratch/mariamma/xraysetu/dataset/covid_binary_test/"
            pathWaImageDirectory = "/scratch/mariamma/xraysetu/dataset/covid_binary_test_whatsapp_CORRECT/"
            if self.whatsapp_data == True:
                self.read_pair_imgs(pathDatasetFile, pathImageDirectory, pathWaImageDirectory)
                return
        self.read_imgs(pathDatasetFile, pathImageDirectory)
        return


    def read_imgs(self, pathDatasetFile, pathImageDirectory):
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
                
                self.listImagePaths.append(imagePath)
                self.listImageLabels.append(imageLabel)  
        fileDescriptor.close()


    def read_pair_imgs(self, pathDatasetFile, pathImageDirectory, pathWaImageDirectory):
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

                imagePathWhatsapp = os.path.join(pathWaImageDirectory, lineItems[0])
                
                self.listImagePaths.append(imagePath)
                self.listImageLabels.append(imageLabel)  
                self.listWAImagePaths.append(imagePathWhatsapp)
        fileDescriptor.close()    