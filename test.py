import os
import cv2
import time
import torch
from SwinIR_model import SwinIR

from torchvision import transforms
from torch.autograd import Variable

from PIL import Image
import numpy as np
from args_fusion import args

Swin_model = SwinIR(in_chans=3, upscale=1, img_size=256,
                        window_size=8, img_range=255., depths=[4, 4, 4, 4],
                        embed_dim=60, num_heads=[4,4,4,4], mlp_ratio=2, upsampler='')
Swin_model.load_state_dict(torch.load(args.resume_Swin))

Swin_model.eval()
# Swin_model.cuda()

from MyDataset import ImagePair

def denorm(mean=[0, 0, 0], std=[1, 1, 1], tensor=None):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def norms(mean=[0, 0, 0], std=[1, 1, 1], *tensors):
    out_tensors = []
    for tensor in tensors:
        for t, m, s in zip(tensor, mean, std):
            t.sub_(m).div_(s)
        out_tensors.append(tensor)
    return out_tensors

def detransformcv2(img, mean=[0, 0, 0], std=[1, 1, 1]):
    img = denorm(mean, std, img).clamp_(0, 1) * 255
    if img.is_cuda:
        img = img.cpu().data.numpy().astype('uint8')
    else:
        img = img.numpy().astype('uint8')
    img = img.transpose([1,2,0])
    return img

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

ir_paths = './images/21_pairs/ir/IR'
vis_paths = './images/21_pairs/vis/VIS'
# ir_paths = './images/21_3chans/IR'
# vis_paths = './images/21_3chans/VIS'
save_path = './outputs/21/'


for i in range(0,21):
    ir = ir_paths + str(i+1) + '.png'
    vis = vis_paths + str(i+1) + '.png'

    pair_loader = ImagePair(impath1=ir, impath2=vis, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]))
    img1, img2 = pair_loader.get_pair()
    img1.unsqueeze_(0)
    img2.unsqueeze_(0)

    with torch.no_grad():
        img1 = Variable(img1, requires_grad=False)
        img2 = Variable(img2, requires_grad=False)
        en1= Swin_model.encoder(img1)
        en2= Swin_model.encoder(img2)
        f = (en1+en2)/2
        res = Swin_model.decoder(f)
        res = denorm(mean, std, res[0]).clamp(0, 1)*255
        res_img = res.cpu().data.numpy().astype('uint8')
        img = res_img.transpose([1,2,0])

    filename = save_path + 'Swin_Fused_' + str(i+1) +'.png'
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = Image.fromarray(img)
    img.save(filename, format='PNG', compress_level=0)


