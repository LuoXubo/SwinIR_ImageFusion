class args():

    # training args
    epochs = 4
    batch_size = 1
    # dataset = 'E:/实验数据/train2014/train2014/'
    # dataset = 'n904@n904:/run/user/1000/gvfs/afp-volume:host=NASGP.local,user=zyk,volume=dataset/COCO_train2014/'
    dataset = '/run/user/1000/gvfs/afp-volume:host=NASGP.local,user=zyk,volume=dataset/COCO_train2014/'

    # dataset_ir = '~/桌面/dataset/KAIST/lwir/'
    dataset_ir = '../dataset/KAIST/lwir/'
    dataset_vi = '../dataset/KAIST/visible/'
    HEIGHT = 256
    WIDTH = 256

    save_model_dir = 'models'
    save_loss_dir = 'models/loss'
    save_rfn_model = 'models/train/fusionnet/'

    img_size = 256
    cuda = 1
    seed = 42
    ssim_weight = [1,10,100,1000,10000]
    ssim_path = ['1e0', '1e1', '1e2', '1e3', '1e4']

    lr = 1e-4
    lr_light = 1e-4
    log_interval = 20

    resume_Swin = './models/final_models/Swin.model'
    resume_fusion_model = None

    model_path_gray = './models/1e2/densefuse_gray.model'
    model_path_rgb = './models/densefuse_rgb.model'

