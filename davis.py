import os
import os.path as osp
import numpy as np
from PIL import Image

import torch
import torchvision
from torch.utils import data
import cv2

import glob
import pdb

class DAVIS_MO_Test(data.Dataset):
    # for multi object, do shuffling

    def __init__(self, root, imset='2017/train.txt', resolution='480p', single_object=False,scale = 480):
        self.scale = scale
        self.root = root
        self.mask_dir = os.path.join(root, 'Annotations', resolution)
        self.mask480_dir = os.path.join(root, 'Annotations', '480p')
        self.image_dir = os.path.join(root, 'JPEGImages', resolution)
        _imset_dir = os.path.join(root, 'ImageSets')
        _imset_f = os.path.join(_imset_dir, imset)

        self.videos = []
        self.num_frames = {}
        self.num_objects = {}
        self.shape = {}
        self.size_480p = {}
        with open(os.path.join(_imset_f), "r") as lines:
            for line in lines:
                _video = line.rstrip('\n')
                self.videos.append(_video)
                self.num_frames[_video] = len(glob.glob(os.path.join(self.image_dir, _video, '*.jpg')))
                _mask = np.array(Image.open(os.path.join(self.mask_dir, _video, '00000.png')).convert("P"))
                self.num_objects[_video] = np.max(_mask)
                self.shape[_video] = np.shape(_mask)
                _mask480 = np.array(Image.open(os.path.join(self.mask480_dir, _video, '00000.png')).convert("P"))
                self.size_480p[_video] = np.shape(_mask480)

        self.K = 11
        self.single_object = single_object

    def __len__(self):
        return len(self.videos)


    def To_onehot(self, mask):
        M = np.zeros((self.K, mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for k in range(self.K):
            M[k] = (mask == k).astype(np.uint8)
        return M
    
    def All_to_onehot(self, masks):
        Ms = np.zeros((self.K, masks.shape[0], masks.shape[1], masks.shape[2]), dtype=np.uint8)
        for n in range(masks.shape[0]):
            Ms[:,n] = self.To_onehot(masks[n])
        return Ms

    def __getitem__(self, index):
        video = self.videos[index]
        info = {}
        info['name'] = video
        info['num_frames'] = self.num_frames[video]
        info['size_480p'] = self.size_480p[video]

        N_frames = np.empty((self.num_frames[video],)+self.shape[video]+(3,), dtype=np.float32)
        N_masks = np.empty((self.num_frames[video],)+self.shape[video], dtype=np.uint8)
        num_objects = torch.LongTensor([int(self.num_objects[video])])
        return num_objects, info

    def load_single_image(self,video,f):
        h,w = self.shape[video]
        if h > w:
            w_ = self.scale
            h_ = int(h/w * self.scale)
        else:
            h_ = self.scale
            w_ = int(w/h * self.scale)

        N_frames = np.empty((1,)+(h_,w_,)+(3,), dtype=np.float32)
        N_masks = np.empty((1,)+(h_,w_,), dtype=np.uint8)        
        # N_frames = np.empty((1,)+self.shape[video]+(3,), dtype=np.float32)
        # N_masks = np.empty((1,)+self.shape[video], dtype=np.uint8)

        img_file = os.path.join(self.image_dir, video, '{:05d}.jpg'.format(f))

        N_frames[0] = cv2.resize(np.array(Image.open(img_file).convert('RGB'))/255.,(w_,h_))
        try:
            mask_file = os.path.join(self.mask_dir, video, '{:05d}.png'.format(f))  
            N_masks[0] = cv2.resize(np.array(Image.open(mask_file).convert('P'), dtype=np.uint8),(w_,h_),interpolation = cv2.INTER_NEAREST)
        except:
            N_masks[0] = 255
        Fs = torch.from_numpy(np.transpose(N_frames.copy(), (3, 0, 1, 2)).copy()).float()
        if self.single_object:
            N_masks = (N_masks > 0.5).astype(np.uint8) * (N_masks < 255).astype(np.uint8)
            Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float()
            num_objects = torch.LongTensor([int(1)])
            return Fs, Ms
        else:
            Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float()
            num_objects = torch.LongTensor([int(self.num_objects[video])])
            return Fs, Ms      

class YOUTUBE_MO_Test(data.Dataset):
    # for multi object, do shuffling

    def __init__(self, root,scale = 480):
        self.scale = scale
        self.root = root
        self.mask_dir = os.path.join(root, 'Annotations')
        self.image_dir = os.path.join(root, 'JPEGImages')

        self.videos = [i.split('/')[-1] for i in glob.glob(self.mask_dir +'/*')]
        self.videos.sort()

        self.shape = {}
        self.num_objects = {}
        for video in self.videos:
            mask_list = glob.glob(os.path.join(self.mask_dir,video,'*'))
            mask_list.sort()
            num_objects = 0
            for mask in mask_list:
                mask_ = np.array(Image.open(mask).convert('P'), dtype=np.uint8)
                self.shape[video] = mask_.shape
                num_objects = max(mask_.max(),num_objects)
            self.num_objects[video] = num_objects


        self.K = 11

    def __len__(self):
        return len(self.videos)


    def To_onehot(self, mask):
        M = np.zeros((self.K, mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for k in range(self.K):
            M[k] = (mask == k).astype(np.uint8)
        return M
    
    def All_to_onehot(self, masks):
        Ms = np.zeros((self.K, masks.shape[0], masks.shape[1], masks.shape[2]), dtype=np.uint8)
        for n in range(masks.shape[0]):
            Ms[:,n] = self.To_onehot(masks[n])
        return Ms

    def __getitem__(self, index):
        video = self.videos[index]
        im_list = glob.glob(os.path.join(self.image_dir,video,'*'))
        im_list.sort()
        mask_list = glob.glob(os.path.join(self.mask_dir,video,'*'))
        mask_list.sort()

        im_list_slt = [i for i in im_list if i.split('/')[-1].split('.')[0] >= mask_list[0].split('/')[-1].split('.')[0]]

        info = {}
        info['name'] = video

        return video,im_list_slt,mask_list

    def load_single_image(self,video,img_file,mask_file):
        h,w = self.shape[video]
        if h > w:
            w_ = self.scale
            h_ = int(h/w * self.scale)
        else:
            h_ = self.scale
            w_ = int(w/h * self.scale)

        N_frames = np.empty((1,)+(h_,w_,)+(3,), dtype=np.float32)
        N_masks = np.empty((1,)+(h_,w_,), dtype=np.uint8)        

        N_frames[0] = cv2.resize(np.array(Image.open(img_file).convert('RGB'))/255.,(w_,h_))
        try:
            N_masks[0] = cv2.resize(np.array(Image.open(mask_file).convert('P'), dtype=np.uint8),(w_,h_),interpolation = cv2.INTER_NEAREST)
            object_list = list(set(N_masks.reshape(-1).tolist()))
            if 0 in object_list:
                object_list.pop(0)
        except:
            N_masks[0] = 255
            object_list = []

        Fs = torch.from_numpy(np.transpose(N_frames.copy(), (3, 0, 1, 2)).copy()).float()
        Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float()
        return Fs, Ms,object_list

if __name__ == '__main__': 

    youtube = YOUTUBE_MO_Test('/smart/haochen/STM/data/youtube/valid/')
    for video,im_list_slt,mask_list in youtube:
        for img_file in im_list_slt:
            img_name = img_file.split('/')[-1].split('.')[0]
            if os.path.join(youtube.mask_dir,video,img_name + '.png') in mask_list:
                mask_file = os.path.join(youtube.mask_dir,video,img_name + '.png')
            else:
                mask_file = ''
            Fs,Ms,object_list = youtube.load_single_image(video,img_file,mask_file)
            print(object_list)
        pdb.set_trace()