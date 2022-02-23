import os
import time
import argparse
from helpers.data_prepare.data_loader import get_loader
from solver import Solver
import torch
#torch.cuda.set_device(3)
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main(config):
    this_time = time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time()))
    '''result_path && log_txt'''
    config.result_path = os.path.join(config.result_path, config.model_type+this_time)
    config.log_txt = os.path.join(config.result_path, this_time)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    '''gt_path'''
    config.gt_path = os.path.join(config.result_path, 'gt')
    if not os.path.exists(config.gt_path):
        os.makedirs(config.gt_path)
    '''log_dir'''
    config.log_dir = os.path.join(config.log_dir, config.model_type+this_time)
    config.log_dir = os.path.join(config.log_dir, 'event')
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    '''model path'''
    config.model_path = os.path.join(config.model_path, config.model_type)
    config.model_path = config.model_path + ' ' + this_time
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)

    train_loader = get_loader(data_path=config.train_path,
                              im_size=config.im_size,
                              batch_size=config.batch_size,    #* len(config.device_ids),
                              num_workers=config.num_workers,
                              kind='train',
                              shu_flag=1,
                              im_flag=config.im_format)
    valid_loader = get_loader(data_path=config.valid_path,
                              im_size=config.im_size,
                              batch_size=config.batch_size,   # * len(config.device_ids),
                              num_workers=config.num_workers,
                              kind='valid',
                              shu_flag=1,
                              im_flag=config.im_format)
    test_loader = get_loader(data_path=config.test_path,
                             im_size=config.im_size,
                             batch_size=config.batch_size,            # * len(config.device_ids),
                             num_workers=config.num_workers,
                             kind='test',
                             shu_flag=0,
                             im_flag=config.im_format)
    solver = Solver(config, train_loader, valid_loader, test_loader)
    solver.train()
    solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    '''For summary'''
    parser.add_argument('--log_dir',     default='./log')
    '''For data_loader'''
    parser.add_argument('--im_size',     default=512)
    parser.add_argument('--im_format', default='png')

    # parser.add_argument('--device_ids',default=[0])
    parser.add_argument('--batch_size',  default=2)

    parser.add_argument('--num_workers', default=32)
    '''Data paths'''

    #guanmai
    # parser.add_argument('--train_path', default='./2DCAS/train/')
    # parser.add_argument('--valid_path', default='./2DCAS/valid/')
    # parser.add_argument('--test_path', default='./2DCAS/test/')
    parser.add_argument('--train_path', default='./check/train-debug/')
    parser.add_argument('--valid_path', default='./check/valid-debug/')
    parser.add_argument('--test_path', default='./check/test/')
#    parser.add_argument('--train_path', default='./picture-yxf/picture-yxf-train/')
#    parser.add_argument('--valid_path', default='./picture-yxf/picture-yxf-valid/')
#    parser.add_argument('--test_path', default='./picture-yxf/picture-yxf-test/')

    '''For training'''
    parser.add_argument('--GPU_id', default=[0])
    parser.add_argument('--epochs',      default=50)
    parser.add_argument('--decay',       default=30)

    parser.add_argument('--lr',          default=1e-4)

    # tversky_loss, gld, bce
#    parser.add_argument('--criterion', default='bce')
    #parser.add_argument('--criterion', default='FocalLoss')
#    parser.add_argument('--criterion', default='DiceLoss')
    parser.add_argument('--criterion', default='DiceLoss')
#    parser.add_argument('--criterion', default='TverskyLoss') 
    parser.add_argument('--focalLoss_a', default=0.8)
    parser.add_argument('--beta1',       default=0.5)  # momentum1 in Adam
    parser.add_argument('--beta2',       default=0.999)
    parser.add_argument('--t',           default=1, help='t for Recurrent step of R2U_Net or R2AttU_Net')

    # U_Net, R2U_Net, AttU_Net, R2AttU_Net, unetpp
    # c_unet, c_att_unet,  deeplab, c3_unet
    #U_Net_ACC,U_Net_CCA,U_Net_d2,U_Net, U_Net_dh2, U_Net_Agg,Nonlocal_Unet5, U_Net_d_CCA,convLSTM
    parser.add_argument('--model_type',  default='U_Net')
    #parser.add_argument('--model_type', default='Nonlocal_Unet5')
#    parser.add_argument('--model_type',  default='R2U_Net')
    parser.add_argument('--model_path',  default='./model_dict')
    parser.add_argument('--result_path', default='./result')

    configuration = parser.parse_args()
    main(configuration)





