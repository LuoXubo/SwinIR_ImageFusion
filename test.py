import os
import torch
from torch.autograd import Variable
from SwinIR_model import SwinIR
import Swin_utils as utils
from args_fusion import args
import numpy as np

def load_model(path_auto, flag_img):
    if flag_img is True:
        nc = 3
    else:
        nc = 1

    input_nc = nc
    output_nc = nc

    depths = [4,4,4,4]
    # Swin_model = SwinIR(in_chans=input_nc, img_size=(256,256), depths=depths, upsampler='')
    Swin_model = SwinIR(in_chans=input_nc, upscale=2, img_size=(256, 256),
                        window_size=8, img_range=225., depths=[4, 4, 4, 4],
                        embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='')
    Swin_model.load_state_dict(torch.load(path_auto))

    Swin_model.eval()
    Swin_model.cuda()
    return Swin_model

def run_demo(Swin_model, infrared_path, visible_path, output_path_root, flag_img, name_ir):
    img_ir, h, w, c = utils.get_test_image(infrared_path, flag=flag_img)
    img_vi, h, w, c = utils.get_test_image(visible_path, flag=flag_img)

    if c == 1:
        if args.cuda:
            img_ir = img_ir.cuda()
            img_vi = img_vi.cuda()

        img_ir = Variable(img_ir, requires_grad=False)
        img_vi = Variable(img_vi, requires_grad=False)

        en_r = Swin_model.encoder(img_ir)
        en_v = Swin_model.encoder(img_vi)

        fusion = (en_v + en_r) / 2

        img_fusion_list = Swin_model.decoder(fusion)

    else:
        img_fusion_blocks = []
        for i in range(c):
            # encoder
            # img_vi_temp = img_vi[:,:,i]
            img_vi_temp = img_vi[:][:][i]
            # img_ir_temp = img_ir[:,:,i]
            img_ir_temp = img_ir[:][:][i]
            if args.cuda:
                img_vi_temp = img_vi_temp.cuda()
                img_ir_temp = img_ir_temp.cuda()
            img_vi_temp = Variable(img_vi_temp, requires_grad=False)
            img_ir_temp = Variable(img_ir_temp, requires_grad=False)

            en_r = Swin_model.encoder(img_ir_temp)
            en_v = Swin_model.encoder(img_vi_temp)
            f = (en_r + en_v)/2
            img_fusion_temp = Swin_model.decoder(f)
            img_fusion_blocks.append(img_fusion_temp)
        img_fusion_list = utils.recons_fusion_images(img_fusion_blocks, h, w)

    output_count=0
    # print(img_fusion_list[0])
    for img_fusion in img_fusion_list:
        file_name = 'fused_' + name_ir
        output_path = output_path_root + file_name
        output_count += 1
        utils.save_image_test(img_fusion, output_path)

def main():
    flag_img = False    # False---Gray
    test_path = './images/21_pairs/ir/'
    path_auto = args.resume_Swin
    output_path_root = './outputs/21/'
    if os.path.exists(output_path_root) is False:
        os.mkdir(output_path_root)

    with torch.no_grad():
        model = load_model(path_auto, True)
        imgs_paths_ir, names = utils.list_images(test_path)
        num = len(imgs_paths_ir)
        for i in range(num):
            name_ir = names[i]
            infrared_path = imgs_paths_ir[i]
            visible_path = infrared_path.replace('ir/', 'vis/')
            if visible_path.__contains__('IR'):
                visible_path = visible_path.replace('IR','VIS')
            else:
                visible_path = visible_path.replace('i.','v.')
            run_demo(model, infrared_path=infrared_path, visible_path=visible_path, output_path_root=output_path_root, name_ir=name_ir, flag_img=flag_img)

        print('Done.....')

if __name__ == '__main__':
    main()











