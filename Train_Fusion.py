from SwinIR_model import SwinIR
from Fusion_Network import FusionBlock_res
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

    upscale = 1
    window_size = 8
    img_size = 256
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    swin_model = SwinIR(in_chans=input_nc, upscale=upscale, img_size=img_size,
                        window_size=window_size, img_range=255., depths=[4,4,4,4],
                        embed_dim=60, num_heads=[4,4,4,4], mlp_ratio=2, upsampler='')

    fusion_model = FusionBlock_res(channels=60)
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    fusion_model.train()

    if args.resume_Swin is not None:
        print('Resuming, initializing using weight from {}.'.format(args.resume_Swin))
        swin_model.load_state_dict(torch.load(args.resume_Swin))
    swin_model.eval()

    optimizer = Adam(fusion_model.parameters(), args.lr)
    mse_loss = torch.nn.MSELoss()
    ssim = pytorch_msssim.msssim

    if args.cuda:
        swin_model.cuda()
        fusion_model.cuda()

    tbar = trange(args.epochs)
    print('Start training.....')

    # creating save path
    temp_path_model = os.path.join(args.save_model_dir, 'Fusion')
    if os.path.exists(temp_path_model) is False:
        os.mkdir(temp_path_model)

    all_ssim_loss = 0.
    all_pixel_loss = 0.

    for e in tbar:
        print('Epoch %d.....'%e)
        # load training database
        image_set_ir, batches = utils.load_dataset(original_imgs_path, batch_size)
        count = 0
        for batch in range(batches):
            image_paths_ir = image_set_ir[batch*batch_size:(batch*batch_size+batch_size)]
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

            x_vi = Variable(img_vi.data.clone(), requires_grad=False)

            ssim_loss_value = 0.
            pixel_loss_value = 0.
            for output in outputs:
                # detail loss
                output = (output - torch.min(output))/(torch.max(output)-torch.min(output))
                output = output*255
                ssim_loss_temp = ssim(output, x_vi, normalize=True)
                ssim_loss_value += 700 * (1 - ssim_loss_temp)

                # feature loss
                g2_ir_fea = en_ir
                g2_vi_fea = en_vi
                g2_fuse_fea = f

                pixel_loss_temp = mse_loss(g2_fuse_fea, 6*g2_ir_fea+3*g2_vi_fea)
                pixel_loss_value += pixel_loss_temp

            ssim_loss_value /= len(outputs)
            pixel_loss_value /= len(outputs)

            total_loss = ssim_loss_value + pixel_loss_value
            total_loss.backward()
            optimizer.step()

            all_ssim_loss += ssim_loss_value.item()
            all_pixel_loss += pixel_loss_value.item()

            if (batch + 1) % args.log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\t mse loss: {:.6f}\t ssim loss: {:.6f}\t total loss: {:.6f}\n".format(
                    time.ctime(), e + 1, count, batches,
                    all_pixel_loss / args.log_interval,
                    all_ssim_loss / 700 / args.log_interval,
                    (all_ssim_loss+all_pixel_loss)/args.log_interval
                )
                tbar.set_description(mesg)
                all_ssim_loss = 0.
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
    torch.save(fusion_model.state_dict(), './models/final_models/Fusion.model')

    print("\nDone, trained model saved at", save_model_path)

if __name__ == '__main__':
    # x1 = torch.randn([1, 3, 256, 256])
    # x2 = torch.randn([1, 3, 256, 256])
    #
    # swin_model = SwinIR(in_chans=3, upscale=1, img_size=256,
    #                     window_size=8, img_range=255., depths=[4, 4, 4, 4],
    #                     embed_dim=60, num_heads=[4, 4, 4, 4], mlp_ratio=2, upsampler='')
    # en1 = swin_model.encoder(x1)
    # en2 = swin_model.encoder(x2)
    # fusion_model = FusionBlock_res(60)
    # f = fusion_model(en1,en2)
    # output = swin_model.decoder(f)
    #
    # ssim = pytorch_msssim.msssim
    # ssim_loss = ssim(output, x1, normalize=True)
    # print(ssim_loss)
    main()



