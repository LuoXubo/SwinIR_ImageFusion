import numpy as np

from SwinIR_model import SwinIR
from Fusion_model import Fusion_network
from tqdm import trange
from torch.optim import Adam
from torch.autograd import Variable
from args_fusion import args
import Swin_utils as utils
import random
import torch
import os
import time
import pytorch_msssim

def main():
    original_imgs_path, names = utils.list_images(args.dataset_ir)
    # print(original_imgs_path)
    train_num = 30000
    original_imgs_path = list(original_imgs_path)
    original_imgs_path = original_imgs_path[:train_num]
    random.shuffle(original_imgs_path)

    img_flag = False # True - RGB, False - gray
    train(original_imgs_path, img_flag)

def train(original_imgs_path, img_flag):
    batch_size = args.batch_size
    in_c = 3
    input_nc = in_c
    output_nc = in_c

    upscale = 4
    window_size = 8
    height = 256
    width = 256
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    swin_model = SwinIR(in_chans=input_nc, upscale=2, img_size=(height, width),
                        window_size=window_size, img_range=225., depths=[4,4,4,4],
                        embed_dim=60, num_heads=[6,6,6,6], mlp_ratio=2, upsampler='')

    fusion_model = Fusion_network(nC=in_c)
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    fusion_model.train()

    if args.resume_Swin is not None:
        print('Resuming, initializing using weight from {}.'.format(args.resume_Swin))
        swin_model.load_state_dict(torch.load(args.resume_Swin))
    swin_model.eval()

    # print(swin_model)
    optimizer = Adam(fusion_model.parameters(), args.lr)
    mse_loss = torch.nn.MSELoss()
    # ssim_loss = pytorch_msssim.msssim

    if args.cuda:
        swin_model.cuda()
        fusion_model.cuda()

    tbar = trange(args.epochs)
    print('Start training.....')

    # creating save path
    temp_path_model = os.path.join(args.save_model_dir, 'fusion_net')
    # temp_path_model = os.path.join(args.save_model_dir, args.ssim_path[i])
    if os.path.exists(temp_path_model) is False:
        os.mkdir(temp_path_model)

    # Loss_pixel = []
    # Loss_ssim = []
    Loss_all = []
    # all_ssim_loss = 0.
    all_pixel_loss = 0.

    for e in tbar:
        print('Epoch %d.....'%e)
        # load training database
        image_set_ir, batches = utils.load_dataset(original_imgs_path, batch_size)
        count = 0
        for batch in range(batches):
            image_paths_ir = image_set_ir[batch*batch_size:(batch*batch_size+batch_size)]
            # print(image_paths_ir)
            img_ir = utils.get_train_images(image_paths_ir, height=args.HEIGHT, width=args.WIDTH, flag=img_flag)

            image_paths_vi = [x.replace('lwir', 'visible') for x in image_paths_ir]
            img_vi = utils.get_train_images(image_paths_vi, height=args.HEIGHT, width=args.WIDTH, flag=img_flag)

            count += 1
            optimizer.zero_grad()
            img_ir = Variable(img_ir, requires_grad=False)
            img_vi = Variable(img_vi, requires_grad=False)

            if args.cuda:
                img_ir = img_ir.cuda()
                img_vi = img_vi.cuda()

            en_ir = swin_model.encoder(img_ir)
            en_vi = swin_model.encoder(img_vi)

            f = fusion_model(en_ir, en_vi)

            outputs = swin_model.decoder(f)

            # x_vi = Variable(img_vi.data.clone(), requires_grad=False)

            # ssim_loss_value = 0.
            pixel_loss_value = 0.
            for output in outputs:
                # detail loss
                output = (output - torch.min(output))/(torch.max(output)-torch.min(output))
                output = output*255
                # ssim_loss_temp = ssim_loss(output, x_vi, normalize=True)
                # ssim_loss_value += 700 * (1 - ssim_loss_temp)

                # feature loss
                g2_ir_fea = en_ir
                g2_vi_fea = en_vi
                g2_fuse_fea = f

                pixel_loss_temp = mse_loss(g2_fuse_fea, 6*g2_ir_fea+3*g2_vi_fea)
                pixel_loss_value += pixel_loss_temp

            # ssim_loss_value /= len(outputs)
            pixel_loss_value /= len(outputs)

            # total_loss = pixel_loss_value
            # total_loss = ssim_loss_value + pixel_loss_value
            # total_loss = pixel_loss_value + args.ssim_weight[i] * ssim_loss_value
            pixel_loss_value.backward()
            optimizer.step()

            # all_ssim_loss += ssim_loss_value.item()
            all_pixel_loss += pixel_loss_value.item()

            if (batch + 1) % args.log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\t mse loss: {:.6f}\t".format(
                    time.ctime(), e + 1, count, batches,
                    all_pixel_loss / args.log_interval,
                    # all_ssim_loss / args.log_interval,
                    # (all_ssim_loss+all_pixel_loss)/args.log_interval
                )
                # mesg = "{}\tEpoch {}:\t[{}/{}]\t pixel loss: {:.6f}\t ssim loss: {:.6f}\t total: {:.6f}".format(
                #     time.ctime(), e + 1, count, batches,
                #                   all_pixel_loss / args.log_interval,
                #                   all_ssim_loss / args.log_interval,
                #                   (args.ssim_weight[i] * all_ssim_loss + all_pixel_loss) / args.log_interval
                # )
                tbar.set_description(mesg)
                # Loss_pixel.append(all_pixel_loss / args.log_interval)
                # Loss_ssim.append(all_ssim_loss / args.log_interval)
                # Loss_all.append((all_pixel_loss+all_ssim_loss) / args.log_interval)
                # Loss_all.append((args.ssim_weight[i] * all_ssim_loss + all_pixel_loss) / args.log_interval)

                # all_ssim_loss = 0.
                all_pixel_loss = 0.
            if (batch+1)%(200*args.log_interval) == 0:
                fusion_model.eval()
                fusion_model.cpu()

                save_model_filename = 'fusion_models/' + "Epoch_" + str(e) + "_iters_" + str(count) + "_" + \
                                      str(time.ctime()).replace(' ', '_').replace(':', '_') + ".model"
                save_model_path = os.path.join(args.save_model_dir, save_model_filename)
                torch.save(swin_model.state_dict(), save_model_path)

                fusion_model.train()
                fusion_model.cuda()
                tbar.set_description('\nCheckpoint, trained model saved ad', save_model_path)

    fusion_model.eval()
    fusion_model.cpu()
    save_model_filename = 'fusion_models/' "Final_epoch_" + str(args.epochs) + "_" + \
                          str(time.ctime()).replace(' ', '_').replace(':', '_') + ".model"
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    torch.save(fusion_model.state_dict(), save_model_path)
    # torch.save(fusion_model.state_dict(), './models/final_models/fusion.model')

    print("\nDone, trained model saved at", save_model_path)

if __name__ == '__main__':
    main()

