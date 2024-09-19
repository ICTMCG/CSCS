
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import PIL
from PIL import Image
import numpy as np
import random
import glob
from random import choice

from torchvision.transforms.transforms import RandomApply, RandomRotation
import albumentations as A
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from tqdm import tqdm

import os
import cv2


class SMSwap_dataset_ssl_Aug(Dataset):
    def __init__(self,
    root_dir,
    img_size=(256,256), gray_channel=False, aug_type='facethin', aug_prob=.5
    ):
        '''
        augtype:  facethin | facechubby
        '''
        self.img_size = img_size
        
        self.gray_channel=gray_channel
        
        self.aug_prob = aug_prob

        self.resize_T = transforms.Resize(size=img_size)
        self.flip = transforms.RandomHorizontalFlip(p=0.5)

        self.norm_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.norm_transform_gray = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.456], [0.224])
        ])

        
        self.root_dir = root_dir
        temp_path   = os.path.join(self.root_dir,'*')
        pathes      = glob.glob(temp_path)
        
        
        self.gt = []
        self.src_gen = []
        self.dst_gen = []
        self.dst_gen_aug = []

        for img_dir in tqdm(pathes):
            img_name = os.path.basename(img_dir)
            try:
                srcs = glob.glob(img_dir+'/*_gensrc.jpg')
                dsts = glob.glob(img_dir+'/*_gendst.jpg')
                if aug_type == 'all':
                    dsts_aug = glob.glob(img_dir+'/*_gendst.jpg_facethin.jpg') + glob.glob(img_dir+'/*_gendst.jpg_facechubby.jpg')
                else:
                    dsts_aug = glob.glob(img_dir+'/*_gendst.jpg_'+aug_type+'.jpg') # facethin | facechubby
            except Exception as e:
                continue
            if len(srcs)==0 or len(dsts)==0:
                continue
            
            self.gt.append(os.path.join(img_dir, img_name))
            self.src_gen.append(srcs)
            self.dst_gen.append(dsts)
            self.dst_gen_aug.append(dsts_aug)
        
    def __len__(self):
        return len(self.gt)

    def __getitem__(self, index):
        
        gt_path = self.gt[index]
        
        src_path = random.choice(self.src_gen[index])
        dst_path = random.choice(self.dst_gen[index])
        dst_aug_pth = None
        if len(self.dst_gen_aug[index])!=0:
            dst_aug_pth = random.choice(self.dst_gen_aug[index])

        gt = self.load_sample(gt_path)
        src = self.load_sample(src_path)
        
        if random.random() < self.aug_prob and dst_aug_pth is not None:
            dst = self.load_sample(dst_aug_pth)
        else:
            dst = self.load_sample(dst_path)

        return gt, src, dst

    def load_sample(self, img_path):

        if self.gray_channel:
            img = Image.open(img_path).convert('L')
        else:
            img = Image.open(img_path).convert('RGB')


        img = self.resize_T(img)

        if self.gray_channel:
            img = self.norm_transform_gray(img)
        else:
            img = self.norm_transform(img)

        return img


class SMSwap_dataset_arcface(Dataset):
    def __init__(self,
    root_dir,
    img_size=(256,256), gray_channel=False, self_recons=False
    ):

        self.self_recons=self_recons

        self.root_dir = root_dir
        self.data = glob.glob(root_dir+'/*.jpg') + glob.glob(root_dir+'/*.png') + glob.glob(root_dir+'/*.jpeg')


        self.img_size = img_size
        
        self.gray_channel=gray_channel

        self.resize_T = transforms.Resize(size=img_size)
        self.flip = transforms.RandomHorizontalFlip(p=0.5)

        self.norm_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])
        self.norm_transform_gray = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.456], [0.224])
        ])



    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        if self.self_recons:
            img1_name = img2_name = random.choice(self.data)
        else:
            img1_name, img2_name = random.sample(self.data, 2)
        img1_path = img1_name
        img2_path = img2_name

        img1 = self.load_sample(img1_path)
        img2 = self.load_sample(img2_path)

        return img1, img2

    def load_sample(self, img_path):

        if self.gray_channel:
            img = Image.open(img_path).convert('L')
        else:
            img = Image.open(img_path).convert('RGB')


        img = self.resize_T(img)

        if self.gray_channel:
            img = self.norm_transform_gray(img)
        else:
            img = self.norm_transform(img)

        return img
    
