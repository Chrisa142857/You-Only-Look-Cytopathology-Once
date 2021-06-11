import glob
import random
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import cv2
from utils.augmentations import horisontal_flip, strong_rand_aug
from torch.utils.data import Dataset
from utils.pysrp.pysrp import Srp
import torchvision.transforms as transforms
from PIL import ImageFile
import matplotlib.pyplot as plt
ImageFile.LOAD_TRUNCATED_IMAGES = True

def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)

def sliding_sample_pt(img_size, overlap, center_box):
    xmin, ymin, xmax, ymax = center_box
    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
    xs = [x for x in range(xmin, xmax, img_size[0]-overlap)]
    ys = [y for y in range(ymin, ymax, img_size[1]-overlap)]
    X, Y = [], []
    for iy in ys:
        for i in range(len(xs)):
            Y += [iy]
    for i in range(len(ys)):
        X += xs
    pts = [[x, y] for x, y in zip(X, Y)]
    return pts

class SrpDataset(Dataset):
    def __init__(
        self, 
        list_path='Z:/wei/PCDD_all/data_sets/sfy1&2_slide_val_list.txt', 
        convert_channel=False, 
        img_size=(1024, 1024), 
        sample_wh=(1024, 1024),
        overlap=0,
        center_side=20000,
        augment=False, 
        multiscale=True, 
        normalized_labels=True, 
        fp_dict={},
        level=0,
        collect_fn=None,
        saved_root = r'A:\sfy1&2_yolos'
        ):
        srp_handle = Srp()
        self.saved_root = saved_root
        self.wh_list = []
        with open(list_path, "r") as file:
            self.srp_files = file.read().split('\n')[:-1]
        # self.srp_files = self.srp_files[::-1]  # [int(len(self.srp_files)*0.5):] [:int(len(self.srp_files)*0.5)][::-1]
        if os.name == 'posix':
            self.srp_files = [f.replace('O:', '/mnt/160_o').replace('\\','/') for f in self.srp_files]
        for idx, srp_file in enumerate(self.srp_files):
            if not srp_file.endswith('.srp'): 
                self.srp_files.remove(srp_file)
                continue
            try:
                srp_handle.open(srp_file)
                h, w = srp_handle.getAttrs()['width'], srp_handle.getAttrs()['height']
            except:
                print(srp_file)
                self.srp_files.remove(srp_file)
                srp_handle.close()
                continue
            if '40x' in srp_file or '40X' in srp_file:
                self.wh_list.append([int(w/2), int(h/2)])
            else:
                self.wh_list.append([w, h])
            srp_handle.close()
        self.img_size = img_size
        self.sample_wh = sample_wh
        self.convert_channel = convert_channel
        self.multiscale = multiscale
        self.batch_count = 0
        self.rand_size_len = 4
        self.level = level
        self.center_side = center_side
        self.overlap = overlap
        self.imgs_info, self.srps_info = collect_fn(self, sample_wh)
        print('Got %d tiles from %s with %d Srp files.' % (len(self.imgs_info), list_path, len(self.srps_info)))

    def sliding_window(self, img_size):
        imgs_info = []
        srps_info = []
        for srp_file, (w, h) in zip(self.srp_files, self.wh_list):
            if not srp_file.endswith('.srp'): continue
            if '40x' in srp_file or '40X' in srp_file:
                level = 1
            else:
                level = 0
            start_pt = [w/2 - self.center_side/2, h/2 - self.center_side/2]
            end_pt = [w/2 + self.center_side/2, h/2 + self.center_side/2]
            center_box = start_pt + end_pt
            LeftTopPts = sliding_sample_pt(img_size, self.overlap, center_box)
            srps_info.append({
                'srp_file': srp_file,
                'num_tiles': len(LeftTopPts)
                })
            imgs_info.extend([{
                'srp_file': srp_file.replace('\\', '/'),
                'name': '%s-%d-%d' % (srp_file.replace('\\', '/').split('/')[-1].split('.srp')[0], int(x/2), int(y/2)),
                'sample_pt': [int(x), int(y), int(img_size[0]), int(img_size[1])],
                'level': level
                } for x, y in LeftTopPts])
        return imgs_info, srps_info

    def img_from_srp(self, srp_file, x, y, w, h, level):

        srp_handle = Srp()
        srp_handle.open(srp_file)
        try:
            img = np.ctypeslib.as_array(srp_handle.ReadRegionRGB(level, y, x, w, h)).reshape((int(h), int(w), 3))
            srp_handle.close()
            img.dtype = np.uint8
        except:
            img = np.zeros((int(h), int(w), 3))
            img.dtype = np.uint8
            img = img[:, :, :3]
        return img

    def __getitem__(self, index):
        img_info = self.imgs_info[index % len(self.imgs_info)]
        if os.path.exists(os.path.join(self.saved_root, img_info['srp_file'].split('/')[-1].split('.srp')[0]+'.pth')):
            print('Skip', img_info['name'])
            return img_info['srp_file'], img_info['name'], img_info['sample_pt'], None, None
        x, y, w, h = img_info['sample_pt']
        img = self.img_from_srp(img_info['srp_file'].rstrip(), x, y, w, h, img_info['level'])
        img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_RGB2BGR) if self.convert_channel else img)  
        targets = None
        img = transforms.ToTensor()(img)
        return img_info['srp_file'], img_info['name'], img_info['sample_pt'], img, targets 

    def __len__(self):
        return len(self.imgs_info)

    def collate_fn(self, batch):

        new_img_size = self.img_size
        img_info1, img_info2, img_info3, imgs, targets = list(zip(*batch))
        if imgs[0] is None: 
            return img_info1, img_info2, img_info3, None, None
        if self.multiscale and self.batch_count % 10 == 0:
            rand_scale = random.choice(range(self.rand_size_len * -1, 1, 1))
            new_img_size = (self.img_size[0] + rand_scale*32, self.img_size[1] + rand_scale*32)
    
        # Resize images to input shape
        imgs = torch.stack([resize(img, new_img_size) for img in imgs])
        self.batch_count += 1
        return img_info1, img_info2, img_info3, imgs, targets


class ListDataset(Dataset):
    def __init__(
        self, 
        list_path='Z:/wei/PCDD_all/data_sets/images1024_valid_new.txt', 
        folder="train_images1024_new", 
        labels_folder="labels", 
        convert_channel=False, 
        need_old_wh=False, 
        img_size=(1024, 1024), 
        augment=False, 
        multiscale=True, 
        normalized_labels=True, 
        fp_dict={}
        ):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = [
            path.replace(folder, labels_folder).replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.rand_size_len = 3
        self.batch_count = 0
        self.fp_dict = fp_dict
        self.convert_channel = convert_channel
        self.need_old_wh = need_old_wh

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = Image.open(img_path).convert('RGB')
        if self.convert_channel:
            img = Image.fromarray(cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR))         
            
        # _, h, w = img.shape
        w, h = img.size

        # ---------
        #  Label
        # ---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        targets = None
        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            boxes = boxes[torch.where(boxes[:, 1]>0)[0], :]
            boxes = boxes[torch.where(boxes[:, 1]<=1)[0], :]
            boxes = boxes[torch.where(boxes[:, 2]>0)[0], :]
            boxes = boxes[torch.where(boxes[:, 2]<=1)[0], :]
            boxes = boxes[torch.where(boxes[:, 3]>0)[0], :]
            boxes = boxes[torch.where(boxes[:, 3]<=1)[0], :]
            boxes = boxes[torch.where(boxes[:, 4]>0)[0], :]
            boxes = boxes[torch.where(boxes[:, 4]<=1)[0], :]

            # Apply augmentations
            if self.augment:
                bboxes_for_aug = [[cx.item(), cy.item(), bw.item(), bh.item()] for cx, cy, bw, bh in boxes[:, 1:]]
                category_for_aug = [box[0].item() for box in boxes]
                img_for_aug = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
                aug_input = {"image": img_for_aug, "bboxes": bboxes_for_aug, "category_id": category_for_aug}
                if w == h:
                    aug_output = strong_rand_aug(p=0.5, width=self.img_size[0], height=self.img_size[1])(**aug_input)
                else:
                    aug_output = strong_rand_aug(no_rotate=True, p=0.5, width=self.img_size[0], height=self.img_size[1])(**aug_input)
                img = aug_output['image']
                img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                if len(bboxes_for_aug) != 0:
                    boxes = torch.cat([torch.FloatTensor([clas, cx, cy, bw, bh]).unsqueeze(0) for clas, (cx, cy, bw, bh ) in zip(aug_output['category_id'], aug_output['bboxes'])])

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(img)

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))
        # if img.shape[1] == 1936:
        #     img = img[:, :-16, :]
        if not self.need_old_wh:
            return img_path, img, targets 
        else:
            return img_path, img, targets, w

    def collate_fn(self, batch):
        if not self.need_old_wh:
            paths, imgs, targets = list(zip(*batch))
        else:
            paths, imgs, targets, ws = list(zip(*batch))
        # Remove empty placeholder targets
        if targets != None:
            targets = [boxes for boxes in targets if boxes is not None]
            # Add sample index to targets
            for i, boxes in enumerate(targets):
                boxes[:, 0] = i
            if targets != []:
                targets = torch.cat(targets, 0)
            else:
                targets = torch.zeros((0, 6))
            # Concat FP
            fps = []
            for b_i, path in enumerate(paths):
                if path not in self.fp_dict:
                    self.fp_dict[path] = torch.FloatTensor(0, 5)
                fps.append(torch.cat((torch.FloatTensor(self.fp_dict[path].shape[0], 1).fill_(b_i), self.fp_dict[path]), 1))
            fps = torch.cat(fps)
            targets = torch.cat((targets, fps))
        new_img_size = self.img_size
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            rand_scale = random.choice(range(self.rand_size_len * -1, 1, 2))
            if self.img_size == (1216, 1936):
                new_img_size = (1216 + rand_scale*64, 1920 + rand_scale*64)
            else:
                new_img_size = (self.img_size[0] + rand_scale*32, self.img_size[1] + rand_scale*32)
    
        # Resize images to input shape
        imgs = torch.stack([resize(img, new_img_size) for img in imgs])

        if imgs.shape[-1] == 1936:
            imgs = imgs[..., :-16]
        self.batch_count += 1
        if not self.need_old_wh:
            return paths, imgs, targets
        else:
            return paths, imgs, targets, ws

    def __len__(self):
        return len(self.img_files)
