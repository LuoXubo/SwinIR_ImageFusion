import os
from os import listdir, mkdir, sep
from os.path import join, exists, splitext
import random
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from args_fusion import args
from scipy.misc import imread, imsave, imresize
# from imageio import imread
import matplotlib as mpl
import cv2
from torchvision import datasets, transforms

def list_images(directory):
    images = []
    names = []
    dir = listdir(directory)
    dir.sort()
    for file in dir:
        name = file.lower()
        images.append(join(directory, file))
        # name1 = name.split('.')
        names.append(name)

    return images, names

def tensor_load_rgbimage(filename, size=None, scale=None, keep_asp=False):
    img = Image.open(filename).convert('RGB')
    if size is not None:
        if keep_asp:
            size2 = int(size * 1.0 / img.size[0] * img.size[1])
            img = img.resize((size, size2), Image.ANTIALIAS)
        else:
            img = img.resize((size, size), Image.ANTIALIAS)

    elif scale is not None:
        img = img.resize((int(img.size[0]/scale), int(img.size[1]/scale)), Image.ANTIALIAS)

    img = np.array(img).transpose(2,0,1)
    img = torch.from_numpy(img).float()
    return img

def tensor_save_rgbimage(tensor, filename, cuda=True):
    if cuda:
        img = tensor.cpu().clamp(0, 255).data[0].numpy()
    else:
        img = tensor.clamp(0, 255).numpy()
    img = img.transpose(1,2,0).astype('uint8')
    img = Image.fromarray(img)
    img.save(filename)

def tensor_save_bgrimage(tensor, filename, cuda=False):
    (b, g, r) = torch.chunk(tensor, 3)
    tensor = torch.cat((r,g,b))
    tensor_save_rgbimage(tensor, filename, cuda)

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w*h)
    features_t = features.transpose(1,2)
    gram = features.bmm(features_t)/(ch*h*w)
    return gram

def matSqrt(x):
    U,D,V = torch.svd(x)
    return U * (D.pow(0.5).diag())*V.t()

def load_dataset(image_path, BATCH_SIZE, num_imgs=None):
    if num_imgs is None:
        num_imgs = len(image_path)
    original_imgs_path = image_path[:num_imgs]

    random.shuffle(original_imgs_path)
    mod = num_imgs % BATCH_SIZE
    print('Batch Size %d.'%BATCH_SIZE)
    print('Train images number %d.'%num_imgs)
    print('Train images samples %s.'%str(num_imgs/BATCH_SIZE))

    if mod > 0:
        print('Train set has been trimmed %d samples...\n'%mod)
        original_imgs_path = original_imgs_path[:-mod]
    batches = int(len(original_imgs_path) // BATCH_SIZE)
    return original_imgs_path, batches

# def get_image(path, height=256, width=256, mode='L'):
#     if mode=='L':
#         image = imread(path, mode=mode)
#     elif mode=='RGB':
#         image = Image.open(path).convert('RGB')
#     if height is not None and width is not None:
#         image = imresize(image, [height, width], interp='nearest')
#     return image

def get_image(path, height=256, width=256, flag=True):
    # print(path)
    if flag is True:
        image = imread(path, mode='RGB')
    else:
        image = imread(path, mode='L')

    if height is not None and width is not None:
        image = imresize(image, [height, width], interp='nearest')
    return image

def get_train_images_auto(paths, height=256, width=256, mode='RGB'):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        if mode == 'RGB':
            flag = True
        else:
            flag = False
        image = get_image(path, height, width, flag=flag)
        # image = get_image(path, height, width, mode=mode)
        if mode == 'L':
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        else:
            image = np.reshape(image, [image.shape[2], image.shape[0], image.shape[1]])

        images.append(image)
        # images = np.append(images, image)
    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    return images

def get_train_images(paths, height=256, width=256, flag=False):
    if isinstance(paths, str):
        paths = list(paths)

    images = []
    for path in paths:
        image = get_image(path, height, width, flag)
        if flag is True:
            image = np.transpose(image, (2,0,1))
        else:
            image = np.reshape(image, [1,height,width])
        images.append(image)

    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    return images


def get_test_image(paths, height=None, width=None, flag=False):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        if flag is True:
            image = imread(path, mode='RGB')
        else:
            image = imread(path, mode='L')
        # get saliency part
        if height is not None and width is not None:
            image = imresize(image, [height, width], interp='nearest')

        base_size = 512
        h = image.shape[0]
        w = image.shape[1]
        c = 1
        if h > base_size or w > base_size:
            c = 4
            if flag is True:
                image = np.transpose(image, (2, 0, 1))
            else:
                image = np.reshape(image, [1, h, w])
            images = get_img_parts(image, h, w)
        else:
            if flag is True:
                image = np.transpose(image, (2, 0, 1))
            else:
                image = np.reshape(image, [1, image.shape[0], image.shape[1]])
            images.append(image)
            images = np.stack(images, axis=0)
            images = torch.from_numpy(images).float()

    return images, h, w, c

def get_img_parts(image, h, w):
    images = []
    h_cen = int(np.floor(h / 2))
    w_cen = int(np.floor(w / 2))
    img1 = image[:, 0:h_cen + 3, 0: w_cen + 3]
    img1 = np.reshape(img1, [1, img1.shape[0], img1.shape[1], img1.shape[2]])
    img2 = image[:, 0:h_cen + 3, w_cen - 2: w]
    img2 = np.reshape(img2, [1, img2.shape[0], img2.shape[1], img2.shape[2]])
    img3 = image[:, h_cen - 2:h, 0: w_cen + 3]
    img3 = np.reshape(img3, [1, img3.shape[0], img3.shape[1], img3.shape[2]])
    img4 = image[:, h_cen - 2:h, w_cen - 2: w]
    img4 = np.reshape(img4, [1, img4.shape[0], img4.shape[1], img4.shape[2]])
    images.append(torch.from_numpy(img1).float())
    images.append(torch.from_numpy(img2).float())
    images.append(torch.from_numpy(img3).float())
    images.append(torch.from_numpy(img4).float())
    return images

def colormap():
    return mpl.colors.LinearSegmentedColormap.from_list('cmap', ['#FFFFFF', '#98F5FF', '#00FF00', '#FFFF00','#FF0000', '#8B0000'], 256)

def save_images(path, data):
    if data.shape[2] == 1:
        data = data.reshape([data.shape[0], data.shape[1]])
    imsave(path, data)

def recons_fusion_images(img_lists, h, w):
    # print(len(img_lists))
    img_f_list = []
    h_cen = int(np.floor(h / 2))
    w_cen = int(np.floor(w / 2))
    c = img_lists[0][0].shape[1]
    ones_temp = torch.ones(1, c, h, w).cuda()
    for i in range(len(img_lists[0])):
        # img1, img2, img3, img4
        img1 = img_lists[0][i]
        img2 = img_lists[1][i]
        img3 = img_lists[2][i]
        img4 = img_lists[3][i]
        # print(img1.shape)
        img_f = torch.zeros(1, c, h, w).cuda()
        count = torch.zeros(1, c, h, w).cuda()

        img_f[:, :, 0:h_cen + 8, 0: w_cen + 8] += img1
        count[:, :, 0:h_cen + 3, 0: w_cen + 3] += ones_temp[:, :, 0:h_cen + 3, 0: w_cen + 3]
        img_f[:, :, 0:h_cen + 3, w_cen - 2: w] += img2
        count[:, :, 0:h_cen + 3, w_cen - 2: w] += ones_temp[:, :, 0:h_cen + 3, w_cen - 2: w]
        img_f[:, :, h_cen - 2:h, 0: w_cen + 3] += img3
        count[:, :, h_cen - 2:h, 0: w_cen + 3] += ones_temp[:, :, h_cen - 2:h, 0: w_cen + 3]
        img_f[:, :, h_cen - 2:h, w_cen - 2: w] += img4
        count[:, :, h_cen - 2:h, w_cen - 2: w] += ones_temp[:, :, h_cen - 2:h, w_cen - 2: w]
        img_f = img_f / count
        img_f_list.append(img_f)
    return img_f_list

def save_image_test(img_fusion, output_path):
    img_fusion = img_fusion.float()
    if args.cuda:
        img_fusion = img_fusion.cpu().data[0].numpy()
        # img_fusion = img_fusion.cpu().clamp(0, 255).data[0].numpy()
    else:
        img_fusion = img_fusion.clamp(0, 255).data[0].numpy()

    img_fusion = (img_fusion - np.min(img_fusion)) / (np.max(img_fusion) - np.min(img_fusion))
    img_fusion = img_fusion * 255
    h, w = img_fusion.shape
    img_fusion = np.array(img_fusion)
    img_fusion.resize([1, h, w])
    img_fusion = img_fusion.transpose(1, 2, 0).astype('uint8')
    # cv2.imwrite(output_path, img_fusion)
    if img_fusion.shape[2] == 1:
        img_fusion = img_fusion.reshape([img_fusion.shape[0], img_fusion.shape[1]])
    # 	img_fusion = imresize(img_fusion, [h, w])
    imsave(output_path, img_fusion)




