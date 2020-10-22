import copy
import cv2
import glob
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
import scipy
import shutil
import skimage
import skimage.transform
import time
import torch
import torch.nn.functional as F
import torchvision
import os
from bisect import bisect_left, bisect_right
from collections import defaultdict, OrderedDict
# from html4vision import Col, imagetable
from PIL import Image
from scipy.ndimage.morphology import distance_transform_edt
from skimage import io
from torch import nn, optim
from torch.autograd import Variable
from torchvision import datasets, transforms, models, utils
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
# from tqdm import tqdm

input_size = 128
output_size = 128
mask_size = 64
# expand_size = (output_size - input_size) // 2
patch_w = output_size // 8
patch_h = output_size // 8
patch = (1, patch_h, patch_w)

def construct_masked(input_img):
    resized = skimage.transform.resize(input_img, (input_size, input_size), anti_aliasing=True)
    result = resized
    resized[(input_size-mask_size)/2:(input_size+mask_size)/2-1,(input_size-mask_size)/2:(input_size+mask_size)/2-1,:] = 1
    # result = np.ones((output_size, output_size))
    # result[expand_size:-expand_size, expand_size:-expand_size, :] = resized
    return result


def blend_result(output_img, input_img, blend_width=8):
    '''
    Blends an input of arbitrary resolution with its output, using the highest resolution of both.
    Returns: final result + source mask.
    '''
    print('Input size:', input_img.shape)
    print('Output size:', output_img.shape)
    in_factor = input_size / output_size
    if input_img.shape[1] < in_factor * output_img.shape[1]:
        # Output dominates, adapt input
        out_width, out_height = output_img.shape[1], output_img.shape[0]
        in_width, in_height = int(out_width * in_factor), int(out_height * in_factor)
        input_img = skimage.transform.resize(input_img, (in_height, in_width), anti_aliasing=True)
    else:
        # Input dominates, adapt output
        in_width, in_height = input_img.shape[1], input_img.shape[0]
        out_width, out_height = int(in_width / in_factor), int(in_height / in_factor)
        output_img = skimage.transform.resize(output_img, (out_height, out_width), anti_aliasing=True)
    
    # Construct source mask
    src_mask = np.zeros((output_size, output_size))
    src_mask[expand_size+1:-expand_size-1, expand_size+1:-expand_size-1] = 1 # 1 extra pixel for safety
    src_mask = distance_transform_edt(src_mask) / blend_width
    src_mask = np.minimum(src_mask, 1)
    src_mask = skimage.transform.resize(src_mask, (out_height, out_width), anti_aliasing=True)
    src_mask = np.tile(src_mask[:, :, np.newaxis], (1, 1, 3))
    
    # Pad input
    input_pad = np.zeros((out_height, out_width, 3))
    x1 = (out_width - in_width) // 2
    y1 = (out_height - in_height) // 2
    input_pad[y1:y1+in_height, x1:x1+in_width, :] = input_img
    
    # Merge
    blended = input_pad * src_mask + output_img * (1 - src_mask)

    print('Blended size:', blended.shape)

    return blended, src_mask


def perform_inpaint(gen_model, input_img, blend_width=8):
    # Enable evaluation mode
    gen_model.eval()
    torch.set_grad_enabled(False)

    # Construct masked input
    input_img = skimage.transform.resize(input_img, (input_size, input_size), anti_aliasing=True)
    i = (input_size - mask_size) // 2
    # i = 50
    masked_img = np.copy(input_img)
    masked_img[i:i+mask_size, i:i+mask_size, :] = 1

    # Convert to torch
    masked_img2 = masked_img.transpose(2, 0, 1)
    masked_img_torch = torch.tensor(masked_img2[np.newaxis], dtype=torch.float)

    # Call generator
    output_img = gen_model(masked_img_torch)

    # Convert to numpy
    output_img = output_img.cpu().numpy()
    output_img = output_img.squeeze().transpose(1, 2, 0)
    output_img = np.clip(output_img, 0, 1)

    # # Blend images
    # norm_input_img = input_img.copy().astype('float')
    # if np.max(norm_input_img) > 1:
    #     norm_input_img /= 255
    # blended_img, src_mask = blend_result(output_img, norm_input_img)
    # blended_img = np.clip(blended_img, 0, 1)

    return input_img, masked_img, output_img

def inference_inpaint(gen_model, masked_img):
    # Convert to torch
    masked_img2 = masked_img.transpose(2, 0, 1)
    masked_img_torch = torch.tensor(masked_img2[np.newaxis], dtype=torch.float)
    # Call generator
    output_img = gen_model(masked_img_torch)
    # Convert to numpy
    output_img = output_img.detach().cpu().numpy()
    output_img = output_img.squeeze().transpose(1, 2, 0)
    output_img = np.clip(output_img, 0, 1)
    return output_img

def load_model(model_path):
    model = CEGenerator(extra_upsample=True)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    # Remove 'module' if present
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module' in k:
            name = k[7:] # remove 'module'
        else:
            name = k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.cpu()
    model.eval()
    return model


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def finish_inpaint(imgs, outputs):
    global output_size, input_size
    result = imgs.clone()
    x1 = (output_size - input_size) // 2
    x2 = x1 + input_size
    y1 = (output_size - input_size) // 2
    y2 = y1 + input_size
    result[:, :, y1:y2, x1:x2] = outputs
    return result


def generate_html(G_net, D_net, device, data_loaders, html_save_path, max_rows=64, outpaint=True):
    '''
    Visualizes one batch from both the training and validation sets.
    Images are stored in the specified HTML file path.
    '''
    G_net.eval()
    D_net.eval()
    torch.set_grad_enabled(False)
    if os.path.exists(html_save_path):
        shutil.rmtree(html_save_path)
    os.makedirs(html_save_path + '/images')

    # Evaluate examples
    for phase in ['train', 'val']:
        imgs, masked_imgs, masked_parts = next(iter(data_loaders[phase]))
        masked_imgs = masked_imgs.to(device)
        outputs = G_net(masked_imgs)
        masked_imgs = masked_imgs.cpu()
        if not(outpaint):
            results = finish_inpaint(imgs, outputs.cpu())
        else:
            results = outputs.cpu()
        # Store images
        for i in range(min(imgs.shape[0], max_rows)):
            save_image(masked_imgs[i], html_save_path + '/images/' + phase + '_' + str(i) + '_masked.jpg')
            save_image(results[i], html_save_path + '/images/' + phase + '_' + str(i) + '_result.jpg')
            save_image(imgs[i], html_save_path + '/images/' + phase + '_' + str(i) + '_truth.jpg')

    # Generate table
    cols = [
        Col('id1', 'ID'),
        Col('img', 'Training set masked', html_save_path + '/images/train_*_masked.jpg'),
        Col('img', 'Training set result', html_save_path + '/images/train_*_result.jpg'),
        Col('img', 'Training set truth', html_save_path + '/images/train_*_truth.jpg'),
        Col('img', 'Validation set masked', html_save_path + '/images/val_*_masked.jpg'),
        Col('img', 'Validation set result', html_save_path + '/images/val_*_result.jpg'),
        Col('img', 'Validation set truth', html_save_path + '/images/val_*_truth.jpg'),
    ]
    imagetable(cols, out_file=html_save_path + '/index.html',
               pathrep=(html_save_path + '/images', 'images'))
    print('Generated image table at: ' + html_save_path + '/index.html')


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
        plt.savefig("./test_dataset.png")
        i += 1
