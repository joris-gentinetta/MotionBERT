import torch
import numpy as np
import ipdb
import glob
import os
import io
import math
import random
import pickle
import math
from torch.utils.data import Dataset, DataLoader
from lib.utils.utils_data import crop_scale

def read_input(keypoints_path, vid_size, scale_range, focus):
    kpts_all = np.load(keypoints_path, allow_pickle=True)
    if vid_size:
        w, h = vid_size
        scale = min(w,h) / 2.0
        kpts_all[:,:,:2] = kpts_all[:,:,:2] - np.array([w, h]) / 2.0
        kpts_all[:,:,:2] = kpts_all[:,:,:2] / scale
        motion = kpts_all
    if scale_range:
        motion = crop_scale(kpts_all, scale_range)
    return motion.astype(np.float32)

class WildDetDataset(Dataset):
    def __init__(self, keypoints_path, clip_len=243, vid_size=None, scale_range=None, focus=None):
        self.keypoints_path = keypoints_path
        self.clip_len = clip_len
        self.vid_all = read_input(keypoints_path, vid_size, scale_range, focus)
        
    def __len__(self):
        'Denotes the total number of samples'
        return math.ceil(len(self.vid_all) / self.clip_len)
    
    def __getitem__(self, index):
        'Generates one sample of data'
        st = index*self.clip_len
        end = min((index+1)*self.clip_len, len(self.vid_all))
        return self.vid_all[st:end]