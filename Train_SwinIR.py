from SwinIR_model import SwinIR
from tqdm import trange
from torch.optim import Adam
from torch.autograd import Variable
from args_fusion import args
import Swin_utils as utils
import random
import torch
import os
import time
# import pytorch_msssim

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    original_imgs_path, names = utils.list_images(args.dataset)
    train_num = 40000
    original_imgs_path = original_imgs_path[:train_num]
    random.shuffle(original_imgs_path)

    train(original_imgs_path)

def train(original_imgs_path):
    batch_size = args.batch_size
    in_c = 3
    if in_c == 1:
        img_model = 'L'
    else:
        img_model = 'RGB'

    input_nc = in_c
    upscale = 1
    window_size = 8

    img_size = 256
    depths = [4,4,4,4]

    swin_model = SwinIR(in_chans=input_nc, upscale=upscale, img_size=img_size,
                        window_size=window_size, img_range=255., depths=depths,
                        embed_dim=60, num_heads=depths, mlp_ratio=2, upsampler='')

    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()

    if args.resume_Swin is not None:
        print('Resuming, initializing using weight from {}.'.format(args.resume_Swin))
        swin_model.load_state_dict(torch.load(args.resume_Swin))
    # print(swin_model)
    optimizer = Adam(swin_model.parameters(), args.lr)
    mse_loss = torch.nn.MSELoss()
    # ssim_loss = pytorch_msssim.msssim

    if args.cuda:
        swin_model.cuda()

    tbar = trange(args.epochs)
    print('Start training.....')

    # creating save path
    temp_path_model = os.path.join(args.save_model_dir, 'Swin/')
    if os.path.exists(temp_path_model) is False:
        os.mkdir(temp_path_model)

    Loss_pixel = []
    Loss_ssim = []
    Loss_all = []
    all_ssim_loss = 0.
    all_pixel_loss = 0.

    for e in tbar:
        print('Epoch %d.....'%e)
        # load training database
        image_set_ir, batches = utils.load_dataset(original_imgs_path, batch_size)
        swin_model.train()
        count = 0
        for batch in range(batches):
            image_paths = image_set_ir[batch*batch_size:(batch*batch_size+batch_size)]
            img = utils.get_train_images_auto(image_paths, height=args.HEIGHT, width=args.WIDTH, mode=img_model)
            count += 1
            optimizer.zero_grad()
            img = Variable(img, requires_grad=False)

            if args.cuda:
                img = img.cuda()
            en = swin_model.encoder(img)
            outputs = swin_model.decoder(en)
            # outputs = swin_model.forward(img)

            x = Variable(img.data.clone(), requires_grad=False)

            # ssim_loss_value = 0.
            pixel_loss_value = 0.
            for output in outputs:
                pixel_loss_temp = mse_loss(output, x)
                # ssim_loss_temp = ssim_loss(output, x, normalize=True)
                # ssim_loss_value += (1-ssim_loss_temp)
                pixel_loss_value += pixel_loss_temp
            # ssim_loss_value /= len(outputs)
            pixel_loss_value /= len(outputs)

            total_loss = pixel_loss_value
            total_loss.backward()
            optimizer.step()

            # all_ssim_loss += ssim_loss_value.item()
            all_pixel_loss += pixel_loss_value.item()

            if (batch + 1) % args.log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\t pixel loss: {:.6f}\t".format(
                    time.ctime(), e + 1, count, batches,
                                  all_pixel_loss / args.log_interval
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
                # Loss_all.append((all_pixel_loss) / args.log_interval)
                # Loss_all.append((args.ssim_weight[i] * all_ssim_loss + all_pixel_loss) / args.log_interval)

                # all_ssim_loss = 0.
                all_pixel_loss = 0.
            if (batch+1)%(200*args.log_interval) == 0:
                swin_model.eval()
                swin_model.cpu()
                save_model_filename = "Epoch_" + str(e) + "_iters_" + str(count) + "_" + \
                                      str(time.ctime()).replace(' ', '_').replace(':', '_') + ".model"
                save_model_path = os.path.join(temp_path_model, save_model_filename)
                torch.save(swin_model.state_dict(), save_model_path)

                swin_model.train()
                swin_model.cuda()
                tbar.set_description('\nCheckpoint, trained model saved ad', save_model_path)

    swin_model.eval()
    swin_model.cpu()
    save_model_filename = "Final_epoch_" + str(args.epochs) + "_" + \
                          str(time.ctime()).replace(' ', '_').replace(':', '_') + ".model"
    save_model_path = os.path.join(temp_path_model, save_model_filename)
    torch.save(swin_model.state_dict(), save_model_path)
    torch.save(swin_model.state_dict(), './models/final_models/Swin.model')

    print("\nDone, trained model saved at", save_model_path)

if __name__ == '__main__':
    main()