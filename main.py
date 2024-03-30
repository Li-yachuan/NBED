import argparse
import os
import torch
import random
import numpy

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--batch_size', default=4, type=int, metavar='BT',
                    help='batch size')
parser.add_argument('--LR', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--weight_decay', '--wd', default=0.0005, type=float,
                    metavar='W', help='default weight decay')
parser.add_argument('--stepsize', default="3", type=str, help='learning rate step size')
parser.add_argument('--maxepoch', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--print_freq', '-p', default=500, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--gpu', default=None, type=str,
                    help='GPU ID')
parser.add_argument('--loss_lmbda', default=None, type=float,
                    help='hype-param of loss 1.1 for BSDS 1.3 for NYUD')
parser.add_argument('--itersize', default=1, type=int,
                    metavar='IS', help='iter size')
parser.add_argument("--encoder", default="Dul-M36",
                    help="caformer-m36,Dul-M36")
parser.add_argument("--decoder", default="unetp",
                    help="unet,unetp,default")
parser.add_argument("--head", default="default",
                    help="default,aspp,atten,cofusion")

parser.add_argument("--savedir", default="tmp")
parser.add_argument("--colorJ", action="store_true")
parser.add_argument("-plr", "--pretrainlr", type=float, default=0.1)
parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
parser.add_argument("--resume", type=str,default=None)
parser.add_argument("--dataset", type=str,default="BSDS")
parser.add_argument("--note", default=None)

args = parser.parse_args()

args.stepsize = [int(i) for i in args.stepsize.split("-")]
print(args.stepsize)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
if args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

random_seed = 3407
random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
numpy.random.seed(random_seed)

import sys
from os.path import join
from data_process import BSDS_Loader,NYUD_Loader,BIPED_Loader,Multicue_Loader
from model.basemodel import Basemodel
from torch.utils.data import DataLoader
from utils import Logger, get_model_parm_nums
from train import train
from test import test,multiscale_test,enhence_test,bright_enhence_test


def main():
    if args.dataset == "BSDS":
        datadir = "/workspace/00Dataset/BSDS-yc"

        train_dataset = BSDS_Loader(root=datadir, split="train",
                                    colorJitter=args.colorJ)
        test_dataset = BSDS_Loader(root=datadir, split="test")
        if args.loss_lmbda is None:
            args.loss_lmbda = 1.1
    elif "NYUD" in args.dataset:
        datadir = "/workspace/00Dataset/NYUD"
        mode = args.dataset.split("-")[1]
        train_dataset = NYUD_Loader(root=datadir, split="train", mode=mode)
        test_dataset = NYUD_Loader(root=datadir, split="test", mode=mode)
        if args.loss_lmbda is None:
            args.loss_lmbda = 1.3
    elif args.dataset == "BIPED":
        datadir = "/workspace/00Dataset/BIPED"
        train_dataset = BIPED_Loader(root=datadir, split="train")
        test_dataset = BIPED_Loader(root=datadir, split="test")
        if args.loss_lmbda is None:
            args.loss_lmbda = 1.1
    elif args.dataset == "BIPEDv2":
        datadir = "/workspace/00Dataset/BIPEDv2"
        # if not os.path.isdir(datadir):
        #     datadir = "/media/aita130/AIDD/PFHan/Pytorch/BIPEDv2"
        # datadir = "/media/aita130/AIDD/PFHan/Pytorch/BIPEDv2"
        train_dataset = BIPED_Loader(root=datadir, split="train")
        test_dataset = BIPED_Loader(root=datadir, split="test")
        if args.loss_lmbda is None:
            args.loss_lmbda = 1.1

    elif 'Multicue' in args.dataset:
        root = "/workspace/00Dataset/multicue_pidinet"
        train_dataset = Multicue_Loader(root=root, split="train", setting=args.dataset.split("-")[1:])
        test_dataset = Multicue_Loader(root=root, split="test", setting=args.dataset.split("-")[1:])
        if args.loss_lmbda is None:
            args.loss_lmbda = 1.1
    elif args.dataset == "UDED":
        datadir = "/workspace/00Dataset/UDED"
        train_dataset = None
        test_dataset = BIPED_Loader(root=datadir, split="test")
        if args.loss_lmbda is None:
            args.loss_lmbda = 1.1
    else:
        raise Exception("Error dataset name")

    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False)

    model = Basemodel(encoder_name=args.encoder,
                      decoder_name=args.decoder,
                      head_name=args.head).cuda()
    print("MODEL SIZE: {}".format(get_model_parm_nums(model)))

    #
    # new_key = 'new_key'
    # if 'old_key' in original_dict:
    #     original_value = original_dict.pop('old_key')  # 移除旧键及其对应的值
    #     original_dict[new_key] = original_value  # 添加新键及原有的值



    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location='cpu')['state_dict']
        ckpt["encoder.conv2.0.weight"] = ckpt.pop("encoder.conv2.1.weight")
        ckpt["encoder.conv2.0.bias"] = ckpt.pop("encoder.conv2.1.bias")
        model.load_state_dict(ckpt)
        print("load pretrained model, successfully!")

    if args.mode == "test":
        assert args.resume is not None
        # test(model, test_loader, save_dir=join(args.savedir,
        #                                        os.path.basename(args.resume).split(".")[0]+"-ss"))
        if "BSDS" in args.dataset.upper():
            # multiscale_test(model,
            #                 test_loader,
            #                 save_dir=join(args.savedir, os.path.basename(args.resume).split(".")[0]+"-ms7"))
            # multiscale_test(model,
            #                 test_loader,
            #                 save_dir=join(args.savedir, os.path.basename(args.resume).split(".")[0] + "-ms3"),
            #                 scale_num=3)
            # enhence_test(model,
            #              test_loader,
            #              save_dir=join(args.savedir, os.path.basename(args.resume).split(".")[0] + "-enh"),)
            bright_enhence_test(model,
                         test_loader,
                         save_dir=join(args.savedir, os.path.basename(args.resume).split(".")[0] + "-brienh"),)



    else:
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, num_workers=args.batch_size, drop_last=True, shuffle=True)

        parameters = {'pretrained.weight': [], 'pretrained.bias': [],
                      'nopretrained.weight': [], 'nopretrained.bias': []}

        for pname, p in model.named_parameters():
            if ("encoder.stages" in pname) or ("encoder.downsample_layers" in pname):
                # p.requires_grad = False
                if "weight" in pname:
                    parameters['pretrained.weight'].append(p)
                else:
                    parameters['pretrained.bias'].append(p)

            else:
                if "weight" in pname:
                    parameters['nopretrained.weight'].append(p)
                else:
                    parameters['nopretrained.bias'].append(p)

        optimizer = torch.optim.Adam([
            {'params': parameters['pretrained.weight'], 'lr': args.LR * args.pretrainlr, 'weight_decay': args.weight_decay},
            {'params': parameters['pretrained.bias'], 'lr': args.LR * 2 * args.pretrainlr, 'weight_decay': 0.},
            {'params': parameters['nopretrained.weight'], 'lr': args.LR * 1, 'weight_decay': args.weight_decay},
            {'params': parameters['nopretrained.bias'], 'lr': args.LR * 2, 'weight_decay': 0.},
        ], lr=args.LR, weight_decay=args.weight_decay)


        # optimizer = torch.optim.Adam(model.parameters(), lr=args.LR, weight_decay=args.weight_decay)
        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.LR,
        #                              weight_decay=args.weight_decay)
        # test(model, test_loader, save_dir=join(args.savedir, 'epoch-init-testing-record'))
        for epoch in range(args.start_epoch, args.maxepoch):
            train(train_loader, model, optimizer, epoch, args)
            test(model, test_loader, save_dir=join(args.savedir, 'epoch-%d-ss-test' % epoch))
            if "BSDS" in args.dataset.upper():
                multiscale_test(model, test_loader, save_dir=join(args.savedir, 'epoch-%d-ms-test' % epoch))
            log.flush()


if __name__ == '__main__':
    import datetime

    # 获取当前日期和时间
    current_time = datetime.datetime.now()
    # 将日期和时间转换为字符串格式
    time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")

    args.savedir = join("output-abl", args.savedir)
    os.makedirs(args.savedir, exist_ok=True)
    log = Logger(join(args.savedir, '%s-log.txt' % (time_string)))
    sys.stdout = log
    cmds = "python"
    for cmd in sys.argv:
        if " " in cmd:
            cmd = "\'" + cmd + "\'"
        cmds = cmds + " " + cmd
    print(cmds)
    print(args)

    main()
