import os
import torch
from torch.autograd import Variable
from SwinIR_model import SwinIR
import Swin_utils as utils
from args_fusion import args
import numpy as np
from scipy.misc import imread, imsave, imresize


Swin_model = SwinIR(in_chans=3, upscale=2, img_size=(256, 256),
                        window_size=8, img_range=225., depths=[4, 4, 4, 4],
                        embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='')
Swin_model.load_state_dict(torch.load(args.resume_Swin))
Swin_model.eval()
Swin_model.cuda()

ir = './IR1.png'
vi = './VIS1.png'

ir_img = utils.get_test_images(ir, )
vi_img = utils.get_test_images(vi)

img_fusion_blocks = []
c=3
h = w = 256

for i in range(c):
    # encoder
    img_vi_temp = vi_img[:,:,i]
    img_ir_temp = ir_img[:,:,i]
    if args.cuda:
        img_vi_temp = img_vi_temp.cuda()
        img_ir_temp = img_ir_temp.cuda()
    img_vi_temp = Variable(img_vi_temp, requires_grad=False)
    img_ir_temp = Variable(img_ir_temp, requires_grad=False)

    en_r = Swin_model.encoder(img_ir_temp)
    en_v = Swin_model.encoder(img_vi_temp)
    f = (en_r + en_v)/2
    img_fusion_temp = Swin_model.decoder_eval(f)
    img_fusion_blocks.append(img_fusion_temp)
img_fusion_list = utils.recons_fusion_images(img_fusion_blocks, h, w)

print(img_fusion_list)
# utils.save_images('./fused.png', im)

