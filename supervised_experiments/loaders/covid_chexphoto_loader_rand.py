import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

from PIL import ImageFile
from torchvision import transforms
import random
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
            tmp_listImagePaths, tmp_listImageLabels, tmp_listWhatsAppPaths,\
                 tmp_listImageNames  =  self.read_imgs(img_file_list, img_dir, "chexphoto")
            self.select_imgs_random(tmp_listImagePaths, tmp_listImageLabels, tmp_listWhatsAppPaths, tmp_listImageNames)
        tmp_listImagePaths, tmp_listImageLabels, tmp_listWhatsAppPaths, tmp_listImageNames  = \
                self.load_covid_images(split)
        self.select_imgs_random(tmp_listImagePaths, tmp_listImageLabels, tmp_listWhatsAppPaths, tmp_listImageNames)        
       
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

    def select_imgs_random(self, tmp_listImagePaths, tmp_listImageLabels, tmp_listWhatsAppPaths, tmp_listImageNames, img_ratio):
        start = 0
        end = len(tmp_listImagePaths)
        num = img_ratio * len(tmp_listImagePaths)
        rand_list = random.sample(range(start, end + 1), num)
        for r in rand_list:
            self.listImagePaths.append(tmp_listImagePaths[r])
            self.listImageLabels.append(tmp_listImageLabels[r])
            self.listWhatsAppPaths.append(tmp_listWhatsAppPaths[r])
            self.listImageNames.append(tmp_listImageNames[r])
        return    


    def read_imgs(self, pathDatasetFile, pathImageDirectory, dataset, pathImageDirectoryWA=None):
        #---- Open file, get image paths and labels
    
        fileDescriptor = open(pathDatasetFile, "r")
        
        #---- get into the loop
        line = True
        chex_cnt = 0
        tmp_listImagePaths = []
        tmp_listImageLabels = []
        tmp_listWhatsAppPaths = []
        tmp_listImageNames = []

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
                        tmp_listImagePaths.append(imagePath)
                        tmp_listImageLabels.append(chexphotoLabels) 
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
                                tmp_listWhatsAppPaths.append(imagePathWhatsApp)
                            if self.image_names == True:
                                tmp_listImageNames.append(lineItems[0])    
                            tmp_listImagePaths.append(imagePath)
                            tmp_listImageLabels.append(covidLabels)  
                    else:
                        if self.whatsapp_data == True:
                            imagePathWhatsApp = os.path.join(pathImageDirectoryWA, lineItems[0])
                            tmp_listWhatsAppPaths.append(imagePathWhatsApp)
                        if self.image_names == True:
                            tmp_listImageNames.append(lineItems[0])    
                        tmp_listImagePaths.append(imagePath)
                        tmp_listImageLabels.append(covidLabels)     
        fileDescriptor.close()
        print("Chexphoto data added = {}".format(chex_cnt))
        return tmp_listImagePaths, tmp_listImageLabels, tmp_listWhatsAppPaths, tmp_listImageNames 


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
                tmp_listImagePaths, tmp_listImageLabels, tmp_listWhatsAppPaths, tmp_listImageNames  = self.read_imgs(pathDatasetFile, pathImageDirectory, "covid", pathWaImageDirectory)
                return tmp_listImagePaths, tmp_listImageLabels, tmp_listWhatsAppPaths, tmp_listImageNames 
        tmp_listImagePaths, tmp_listImageLabels, tmp_listWhatsAppPaths, tmp_listImageNames  = self.read_imgs(pathDatasetFile, pathImageDirectory, "covid")
        return tmp_listImagePaths, tmp_listImageLabels, tmp_listWhatsAppPaths, tmp_listImageNames 