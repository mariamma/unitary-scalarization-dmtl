import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

from PIL import ImageFile
from torchvision import transforms
ImageFile.LOAD_TRUNCATED_IMAGES = True

#-------------------------------------------------------------------------------- 

class CovidChexphotoDatasetGenerator(Dataset):
    
    #-------------------------------------------------------------------------------- 
    
    def __init__ (self, img_dir, img_file_list, img_size, chexphoto_labels, split,
                whatsapp_data=False, image_names=False, covid_only=False):
    
        self.listImagePaths = []
        self.listImageLabels = []
        self.listImageNames = []
        self.listWhatsAppPaths = []
        self.chexphoto_labels = chexphoto_labels
        self.whatsapp_data = whatsapp_data
        self.image_names = image_names
        self.covid_only = covid_only


        self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(size=img_size),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        if self.covid_only == False:                                        
            self.read_imgs(img_file_list, img_dir, "chexphoto")
        self.load_covid_images(split)
       
    #-------------------------------------------------------------------------------- 
    
    def __getitem__(self, index):
        
        imagePath = self.listImagePaths[index]
        
        imageData = Image.open(imagePath).convert('RGB')
        imageLabel= torch.FloatTensor(self.listImageLabels[index])
       
        if self.transform != None: imageData = self.transform(imageData)

        if self.whatsapp_data == True:
            WAimagePath = self.listWhatsAppPaths[index]
            WAImgData = Image.open(WAimagePath).convert('RGB')
            if self.transform != None: WAImgData = self.transform(WAImgData)

            if self.image_names == True:
                image_name = self.listImageNames[index]
                return  image_name, imageData, WAImgData, imageLabel 
            else:    
                return imageData, WAImgData, imageLabel

        return imageData, imageLabel
        
    #-------------------------------------------------------------------------------- 
    
    def __len__(self):
        
        return len(self.listImagePaths)
    
    #-------------------------------------------------------------------------------- 
    
    def read_imgs(self, pathDatasetFile, pathImageDirectory, dataset, pathImageDirectoryWA=None):
        #---- Open file, get image paths and labels
    
        fileDescriptor = open(pathDatasetFile, "r")
        
        #---- get into the loop
        line = True
        chex_cnt = 0

        while line:
                
            line = fileDescriptor.readline()
            line = line.strip()
            
            #--- if not empty
            if line:
                if dataset == "chexphoto":
                    lineItems = line.split(",")
                else:    
                    lineItems = line.split(" ")
                
                imagePath = os.path.join(pathImageDirectory, lineItems[0])
                
                imageLabel = lineItems[1:]
                imageLabel = [int(float(i)) for i in imageLabel]
            
                if dataset == "chexphoto":
                    chexphotoLabels = [0] * 15
                    for x in self.chexphoto_labels:
                        chexphotoLabels[x] = imageLabel[x]
                    if sum(chexphotoLabels) > 0:
                        self.listImagePaths.append(imagePath)
                        self.listImageLabels.append(chexphotoLabels) 
                        chex_cnt += 1 
                else:
                    covidLabels = [0] * 15
                    if imageLabel[6] == 1:
                        covidLabels[7] = 1
                    elif imageLabel[14] == 1:                                 
                        covidLabels[0] = 1
                    else:
                        covidLabels[14] = 1   
                        
                    if self.covid_only == True: 
                        if covidLabels[14] == 1:     
                            if self.whatsapp_data == True:
                                imagePathWhatsApp = os.path.join(pathImageDirectoryWA, lineItems[0])
                                self.listWhatsAppPaths.append(imagePathWhatsApp)
                            if self.image_names == True:
                                self.listImageNames.append(lineItems[0])    
                            self.listImagePaths.append(imagePath)
                            self.listImageLabels.append(covidLabels)  
                    else:
                        if self.whatsapp_data == True:
                            imagePathWhatsApp = os.path.join(pathImageDirectoryWA, lineItems[0])
                            self.listWhatsAppPaths.append(imagePathWhatsApp)
                        if self.image_names == True:
                            self.listImageNames.append(lineItems[0])    
                        self.listImagePaths.append(imagePath)
                        self.listImageLabels.append(covidLabels)     
        fileDescriptor.close()
        print("Chexphoto data added = {}".format(chex_cnt))
        

    def load_covid_images(self, split):
        covid_pathDirData = "/scratch/mariamma/xraysetu/dataset/covidx3_upsampled/"
        covid_lugsegdir = "/data/mariammaa/dataset/covidx3_upsampled_lung_segment/"
        if split == "train":
            pathDatasetFile = "/scratch/mariamma/xraysetu/dataset/covid_train.txt"
            pathImageDirectory = os.path.join(covid_pathDirData,"train") 
            lungSegDir = os.path.join(covid_lugsegdir, "train")
        elif split == "val":
            pathDatasetFile = "/scratch/mariamma/xraysetu/dataset/covid_val.txt"
            pathImageDirectory = os.path.join(covid_pathDirData,"test")
            lungSegDir = os.path.join(covid_lugsegdir, "val")
        elif split == "test":    
            pathDatasetFile = "/scratch/mariamma/xraysetu/dataset/covid_test.txt"
            pathImageDirectory = "/scratch/mariamma/xraysetu/dataset/covid_binary_test/"
            pathWaImageDirectory = "/scratch/mariamma/xraysetu/dataset/covid_binary_test_whatsapp_CORRECT/"
            lungSegDir = os.path.join(covid_lugsegdir, "test")
            # lung_seg_dir = "/scratch/mariamma/xraysetu/dataset/covid_binary_test_lung_segments/"
            if self.whatsapp_data == True:
                self.read_imgs(pathDatasetFile, pathImageDirectory, "covid", pathWaImageDirectory)
                return
        self.read_imgs(pathDatasetFile, pathImageDirectory, "covid")
        return