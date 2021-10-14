"""
Author: Isabella Liu 7/18/21
Feature: Load data from messy-table-dataset
"""

import os
import numpy as np
import random
from PIL import Image
import torch
import torchvision.transforms as Transforms
from torch.utils.data import Dataset

from utils.config import cfg
from utils.util import load_pickle


class MessytableDataset(Dataset):
    def __init__(self, split_file, gaussian_blur=False, color_jitter=False, debug=False, sub=100, isReal=False):
        """
        :param split_file: Path to the split .txt file, e.g. train.txt
        :param gaussian_blur: Whether apply gaussian blur in data augmentation
        :param color_jitter: Whether apply color jitter in data augmentation
        :param debug: Debug mode, load less data
        :param sub: If debug mode is enabled, sub will be the number of data loaded
        """
        self.img_L, self.img_R, self.img_depth_l, self.img_depth_r, self.img_meta, self.img_sim = \
            self.__get_split_files__(split_file, debug, sub, isTest=False, onReal=isReal)
        self.gaussian_blur = gaussian_blur
        self.color_jitter = color_jitter
        self.real = isReal

    @staticmethod
    def __get_split_files__(split_file, debug=False, sub=100, isTest=False, onReal=False):
        """
        :param split_file: Path to the split .txt file, e.g. train.txt
        :param debug: Debug mode, load less data
        :param sub: If debug mode is enabled, sub will be the number of data loaded
        :param isTest: Whether on test, if test no random shuffle
        :param onReal: Whether test on real dataset, folder and file names are different
        :return: Lists of paths to the entries listed in split file
        """
        dataset = cfg.REAL.DATASET if onReal else cfg.DIR.DATASET
        img_left_name = cfg.REAL.LEFT if onReal else cfg.SPLIT.LEFT
        img_right_name = cfg.REAL.RIGHT if onReal else cfg.SPLIT.RIGHT

        with open(split_file, 'r') as f:
            prefix = [line.strip() for line in f]
            if isTest is False:
                np.random.shuffle(prefix)

            img_L = [os.path.join(dataset, p, img_left_name) for p in prefix]
            img_R = [os.path.join(dataset, p, img_right_name) for p in prefix]

            depth_dir = cfg.REAL.DEPTHPATH if onReal else cfg.DIR.DATASET
            
            img_depth_l = [os.path.join(depth_dir, p, cfg.SPLIT.DEPTHL) for p in prefix]
            img_depth_r = [os.path.join(depth_dir, p, cfg.SPLIT.DEPTHR) for p in prefix]
            img_meta = [os.path.join(depth_dir, p, cfg.SPLIT.META) for p in prefix]
            #img_label = [os.path.join(depth_dir, p, cfg.SPLIT.LABEL) for p in prefix]

            if debug is True:
                img_L = img_L[:sub]
                img_R = img_R[:sub]
                img_depth_l = img_depth_l[:sub]
                img_depth_r = img_depth_r[:sub]
                img_meta = img_meta[:sub]
                #img_label = img_label[:sub]

        # If training with pix2pix + cascade, load real dataset as input to the discriminator
        if isTest is False:
            img_real_list = cfg.SPLIT.TRAIN
            with open(img_real_list, 'r') as f:
                prefix = [line.strip() for line in f]
                img_sim_L = [os.path.join(cfg.DIR.DATASET, p, cfg.SPLIT.LEFT) for p in prefix]
                img_sim_R = [os.path.join(cfg.DIR.DATASET, p, cfg.SPLIT.RIGHT) for p in prefix]
                img_sim = img_sim_L + img_sim_R
            np.random.shuffle(img_sim)
            return img_L, img_R, img_depth_l, img_depth_r, img_meta, img_sim
        else:
            return img_L, img_R, img_depth_l, img_depth_r, img_meta#, img_label

    @staticmethod
    def __data_augmentation__(gaussian_blur=False, color_jitter=False):
        """
        :param gaussian_blur: Whether apply gaussian blur in data augmentation
        :param color_jitter: Whether apply color jitter in data augmentation
        Note:
            If you want to change the parameters of each augmentation, you need to go to config files,
            e.g. configs/remote_train_config.yaml
        """
        transform_list = [
            Transforms.ToTensor()
        ]
        if gaussian_blur:
            gaussian_sig = random.uniform(cfg.DATA_AUG.GAUSSIAN_MIN, cfg.DATA_AUG.GAUSSIAN_MAX)
            transform_list += [
                Transforms.GaussianBlur(kernel_size=cfg.DATA_AUG.GAUSSIAN_KERNEL, sigma=gaussian_sig)
            ]
        if color_jitter:
            bright = random.uniform(cfg.DATA_AUG.BRIGHT_MIN, cfg.DATA_AUG.BRIGHT_MAX)
            contrast = random.uniform(cfg.DATA_AUG.CONTRAST_MIN, cfg.DATA_AUG.CONTRAST_MAX)
            transform_list += [
                Transforms.ColorJitter(brightness=[bright, bright],
                                       contrast=[contrast, contrast])
            ]
        # Normalization
        #transform_list += [
        #    Transforms.Normalize(
        #        mean=[0.485, 0.456, 0.406],
        #        std=[0.229, 0.224, 0.225],
        #    )
        #]
        custom_augmentation = Transforms.Compose(transform_list)
        return custom_augmentation

    def __len__(self):
        return len(self.img_L)

    def __getitem__(self, idx):
        process = self.__data_augmentation__(self.gaussian_blur, self.color_jitter)
        #print(self.img_L, np.array(Image.open(self.img_L[idx]).convert('RGB')).shape)
        #print(np.array(Image.open(self.img_L[idx])).shape, np.array(Image.open(self.img_L[idx]).convert('RGB')).shape, np.array(Image.open(self.img_L[idx]).convert('RGB').resize((540,960))).shape)
        #print(np.array(Image.open(self.img_L[idx])).shape)  
        img_L_rgb = Image.open(self.img_L[idx]).convert('RGB')   # [H, W, 1], in (0, 1)
        img_R_rgb = Image.open(self.img_R[idx]).convert('RGB')
        L_a = np.array(img_L_rgb)
        R_a = np.array(img_R_rgb)
        #print(L_a[0,0,1]==R_a[0,0,4])
        
        #print(img_L_rgb.shape)
        img_depth_l = np.array(Image.open(self.img_depth_l[idx])) / 1000    # convert from mm to m
        img_depth_r = np.array(Image.open(self.img_depth_r[idx])) / 1000    # convert from mm to m
        img_meta = load_pickle(self.img_meta[idx])
        #print('other ', img_depth_l.shape)

        # For unpaired pix2pix, load a random real image from real dataset [H, W, 1], in value range (-1, 1)
        #print(np.array(Image.open(random.choice(self.img_sim)).convert('RGB')).shape)
        img_sim_rgb = Image.open(random.choice(self.img_sim)).convert('RGB')
        #print(img_sim_rgb.shape)

        #img_L_rgb, img_R_rgb, img_sim_rgb = process(img_L_rgb), process(img_R_rgb), process(img_sim_rgb)
        #print(img_L_rgb.shape, img_R_rgb.shape, img_sim_rgb.shape)

        # Convert depth map to disparity map
        extrinsic_l = img_meta['extrinsic_l']
        extrinsic_r = img_meta['extrinsic_r']
        intrinsic_l = img_meta['intrinsic_l']
        baseline = np.linalg.norm(extrinsic_l[:, -1] - extrinsic_r[:, -1])
        focal_length = intrinsic_l[0, 0] / 2

        mask = img_depth_l > 0
        img_disp_l = np.zeros_like(img_depth_l)
        img_disp_l[mask] = focal_length * baseline / img_depth_l[mask]
        mask = img_depth_r > 0
        img_disp_r = np.zeros_like(img_depth_r)
        img_disp_r[mask] = focal_length * baseline / img_depth_r[mask]

        img_L_rgb, img_R_rgb, img_sim_rgb = process(img_L_rgb), process(img_R_rgb), process(img_sim_rgb)
        img_L_rgb = (img_L_rgb - 0.5) / 0.5
        img_R_rgb = (img_R_rgb - 0.5) / 0.5
        img_sim_rgb = (img_sim_rgb - 0.5) / 0.5
        # random crop the image to 256 * 512
        #print(img_L_rgb.shape, img_R_rgb.shape, img_sim_rgb.shape)
        h, w = 540, 960
        th, tw = cfg.ARGS.CROP_HEIGHT, cfg.ARGS.CROP_WIDTH
        x = random.randint(0, h - th)
        y = random.randint(0, w - tw)
        #print(img_L_rgb.shape, img_R_rgb.shape, img_sim_rgb.shape, img_disp_l.shape, h, w, th, tw, x, y)
        if self.real:
            img_L_rgb = img_L_rgb[:,2*x: 2*(x+th), 2*y: 2*(y+tw)]
            img_R_rgb = img_R_rgb[:,2*x: 2*(x+th), 2*y: 2*(y+tw)]
        else:
            img_L_rgb = img_L_rgb[:,x:(x+th), y:(y+tw)]
            img_R_rgb = img_R_rgb[:,x:(x+th), y:(y+tw)]
        img_disp_l = img_disp_l[2*x: 2*(x+th), 2*y: 2*(y+tw)]  # depth original res in 1080*1920
        img_depth_l = img_depth_l[2*x: 2*(x+th), 2*y: 2*(y+tw)]
        img_disp_r = img_disp_r[2*x: 2*(x+th), 2*y: 2*(y+tw)]
        img_depth_r = img_depth_r[2*x: 2*(x+th), 2*y: 2*(y+tw)]
        img_sim_rgb = img_sim_rgb[:,x: (x+th), y: (y+tw)]  # real original res in 1080*1920

        #img_L_rgb, img_R_rgb, img_sim_rgb = process(img_L_rgb), process(img_R_rgb), process(img_sim_rgb)
        #print(img_L_rgb.shape, img_R_rgb.shape, img_disp_l.shape, img_depth_l.shape, img_sim_rgb.shape)
        

        item = {}

        item['img_L'] = img_L_rgb  # [bs, 1, H, W]
        item['img_R'] = img_R_rgb  # [bs, 1, H, W]
        item['img_sim'] = img_sim_rgb  # [bs, 3, 2*H, 2*W]
        item['img_disp_l'] = torch.tensor(img_disp_l, dtype=torch.float32).unsqueeze(0)  # [bs, 1, H, W] in dataloader
        item['img_depth_l'] = torch.tensor(img_depth_l, dtype=torch.float32).unsqueeze(0)  # [bs, 1, H, W]
        item['img_disp_r'] = torch.tensor(img_disp_r, dtype=torch.float32).unsqueeze(0)  # [bs, 1, H, W]
        item['img_depth_r'] = torch.tensor(img_depth_r, dtype=torch.float32).unsqueeze(0)  # [bs, 1, H, W]
        item['prefix'] = self.img_L[idx].split('/')[-2]
        item['focal_length'] = torch.tensor(focal_length, dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        item['baseline'] = torch.tensor(baseline, dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        
        return item


if __name__ == '__main__':
    cdataset = MessytableDataset(cfg.SPLIT.TRAIN)
    item = cdataset.__getitem__(0)
    print(item['img_L'].shape)
    print(item['img_R'].shape)
    print(item['img_disp_l'].shape)
    print(item['prefix'])
    print(item['img_real'].shape)
