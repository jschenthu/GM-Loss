import argparse
import logging
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data.distributed
from torch.nn.utils import clip_grad_norm_
from eval import verification

import losses
from backbones import get_model
from dataset import MXFaceDataset, SyntheticDataset, DataLoaderX
from partial_fc_gm import PartialFC
from utils.utils_amp import MaxClipGradScaler
from utils.utils_callbacks import CallBackVerification, CallBackLogging, CallBackModelCheckpoint
from utils.utils_config import get_config
from utils.utils_logging import AverageMeter, init_logging


def main(args):
    weight = torch.load(args.prefix)
    resnet = get_model(args.network, dropout=0, fp16=False).cuda()
    resnet.load_state_dict(weight)
    resnet.eval()

    ver_list = []
    ver_name_list = []
    val_targets = ["lfw", "cfp_fp", "agedb_30"]
    #val_targets = ["lfw"]

    for name in val_targets:
        path = os.path.join("train_tmp/faces_vgg_112", name + ".bin")
        if os.path.exists(path):
            data_set = verification.load_bin(path, (112, 112))
            ver_list.append(data_set)
            ver_name_list.append(name)


    for i in range(len(ver_list)):
        acc1, std1, acc2, std2, xnorm, embeddings_list, acc3, std3 = verification.test(
            ver_list[i], resnet, 10, 10, pca=0)
        print('[%s]XNorm: %f' % (ver_name_list[i], xnorm))
        print('[%s]Accuracy-Train: %1.5f+-%1.5f' % (ver_name_list[i], acc1, std1))
        print('[%s]Accuracy-Flip: %1.5f+-%1.5f' % (ver_name_list[i], acc2, std2))
        print('[%s]Thresholds: %1.5f+-%1.5f' % (ver_name_list[i], acc3, std3))

    for i in range(len(ver_list)):
        acc1, std1, acc2, std2, xnorm, embeddings_list, acc3, std3 = verification.test(
            ver_list[i], resnet, 10, 10, pca=64)
        print('[%s]XNorm: %f' % (ver_name_list[i], xnorm))
        print('[%s]Accuracy-Train: %1.5f+-%1.5f' % (ver_name_list[i], acc1, std1))
        print('[%s]Accuracy-Flip: %1.5f+-%1.5f' % (ver_name_list[i], acc2, std2))
        print('[%s]Thresholds: %1.5f+-%1.5f' % (ver_name_list[i], acc3, std3))


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Testing')
    parser.add_argument('--prefix', default='work_dirs/webface_r50/backbone_26824.pth', help='path to load model.')
    parser.add_argument('--network', default='r50', type=str, help='')
    main(parser.parse_args())
