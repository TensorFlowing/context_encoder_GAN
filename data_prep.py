import numpy as np
import matplotlib.pyplot as plt
import glob
import time
import torch
import torch.nn.functional as F
import torchvision
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models, utils

def create_circular_mask(h_input, w_input, center=None, radius=None):
    if center is None: # use the middle of the image
        center = (int(h_input/2), int(w_input/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w_input-center[0], h_input-center[1])
    Y, X = np.meshgrid(np.linspace(start=0, stop=h_input, num=h_input), np.linspace(start=0, stop=w_input, num=w_input))
    dist_from_center = np.sqrt((X-center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center >= radius
    mask = np.stack([mask,mask,mask],axis=2)
    return mask

def create_rect_mask(h_input, w_input, center, h_rect, w_rect):
    Y, X = np.meshgrid(np.linspace(start=0, stop=h_input, num=h_input), np.linspace(start=0, stop=w_input, num=w_input))
    mask = 1-(np.abs(Y-center[0])<h_rect/2)*(np.abs(X-center[1])<w_rect/2)
    mask = np.stack([mask,mask,mask],axis=2)
    return mask

class CEImageDataset(Dataset):
    def __init__(self, root, transform, input_shape=(256,256)):
        self.transform = transform
        self.input_shape = input_shape
        self.files = list(glob.glob(root + "/**/*.jpg", recursive=True))

    def genRandomMask(self, input_shape):
        # Generate a random value (0~1)
        chance = np.random.rand()
        (h,w) = input_shape        
        if (chance<0.5):
            # Generate a circular 
            center = (np.random.rand()*h, np.random.rand()*w)
            radius = 0.5*np.random.rand()*np.amin([h,w])
            mask = create_circular_mask(h, w, center, radius)
        else:
            # Generate a rectangular mask
            center = (np.random.rand()*h, np.random.rand()*w)
            (h_rect, w_rect) = (np.random.rand()*h*0.5, np.random.rand()*w*0.5)
            mask = create_rect_mask(h, w, center, h_rect, w_rect)
        # Convert numpy to torch
        mask = torch.tensor(mask.transpose(2, 0, 1), dtype=torch.float32)
        return mask

    def apply_center_mask(self, img):
        mask = self.genRandomMask(self.input_shape) # masked region=0, other=1
        masked_img = img.clone()
        # print("mask.size()=", mask.size())
        # print("masked_img.size()=", masked_img.size())
        masked_img[mask==0] = 0
        return mask, masked_img 

    def __getitem__(self, index):
        try:
            img = Image.open(self.files[index % len(self.files)]).convert('RGB')
            img = self.transform(img)
        except:
            # Likely corrupt image file, so generate black instead
            img = tensor.zeros((3, self.input_shape[0], self.input_shape[1]))
        mask, masked_img = self.apply_center_mask(img)
        return img, mask, masked_img

    def __len__(self):
        return len(self.files)


if __name__ == "__main__":
    train_dir = "/mnt/Data/yangbo/data/Places365/train_large/"
    input_size = 256
    my_tf = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])
    dataset = CEImageDataset(train_dir, my_tf, input_shape=(256,256))
    i = 0
    while i<100:
        img, mask, masked_img = dataset[i]
        img_np = img.numpy().transpose((1,2,0))
        mask = mask.numpy()[0,:,:]
        masked_img_np = masked_img.numpy().transpose((1,2,0))
        plt.subplot(131)
        plt.imshow(img_np)
        plt.subplot(132)
        plt.imshow(mask)
        plt.subplot(133)
        plt.imshow(masked_img_np)
        plt.savefig("./test_dataset2.png")
        i += 1
