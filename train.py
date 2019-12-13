import argparse
import time
import csv
import datetime
import os

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn as nn
import torch.utils.data

from path import Path
from itertools import chain
from tensorboardX import SummaryWriter
from tqdm import tqdm
import custom_transforms

import models
from sequence_folders import Generate_train_set, Generate_val_set
from utils import  max_normalize, mean_normalize, tensor2array, save_checkpoint


parser = argparse.ArgumentParser(description='DeepFusion',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset_dir", metavar='DIR',
                    help='path to pre-processed dataset')
parser.add_argument("--label_dir", metavar='DIR',
                    help='path to pre-processed label')
parser.add_argument('--pretrained-model', dest='pretrained_model', default=None, metavar='PATH',
                    help='path to pre-trained model')
parser.add_argument('--name', dest='name', type=str, default='demo', required=True,
                    help='name of the experiment, checpoints are stored in checpoints/name')

parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch-size', default=0, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if not set)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=2e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                    help='beta parameters for adam')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('--FCCMnet', dest='FCCMnet', type=str, default='Baseline', help='depth network architecture.')

parser.add_argument('--normalization', dest='normalization', type=str, default='mean', help='normalization method.')


def main():
    global args, best_error, n_iter
    n_iter = 0
    best_error = 0

    args = parser.parse_args()
    save_path = Path(args.name)
    args.save_path = 'checkpoints'/save_path 
    print('=> will save everything to {}'.format(args.save_path))
    args.save_path.makedirs_p()

    train_writer = SummaryWriter(args.save_path)
    torch.manual_seed(args.seed)



    train_transform = custom_transforms.Compose([
          custom_transforms.RandomRotate(),
          custom_transforms.RandomHorizontalFlip(),
          custom_transforms.RandomScaleCrop(),
          custom_transforms.ArrayToTensor() ])

    train_set = Generate_train_set(
        root = args.dataset_dir,
        label_root = args.label_dir,
        transform=train_transform,
        seed=args.seed,
        train=True
    )

    val_set = Generate_val_set(
        root = args.dataset_dir,
        label_root = args.label_dir,
        seed=args.seed
    )

    print('{} samples found in {} train scenes'.format(len(train_set), len(train_set.scenes)))
    print('{} samples found in {} val scenes'.format(len(val_set), len(val_set.scenes)))
    
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    print("=> creating model")
    FCCM_net = getattr(models, args.FCCMnet)().cuda()

    if args.pretrained_model:
        print("=> using pre-trained weights for net")
        weights = torch.load(args.pretrained_model)
        FCCM_net.load_state_dict(weights['state_dict'])
    else:
        FCCM_net.init_weights()

    cudnn.benchmark = True
    FCCM_net = torch.nn.DataParallel(FCCM_net)

    print('=> setting adam solver')

    parameters = chain(FCCM_net.parameters())
    optimizer = torch.optim.Adam(parameters, args.lr,
                                 betas=(args.momentum, args.beta),
                                 weight_decay=args.weight_decay)


    is_best = False
    best_error = float("inf") 
    FCCM_net.train()

    for epoch in tqdm(range(args.epochs)):
        
        loss = train(train_loader, FCCM_net, optimizer, args.epoch_size, train_writer)
        validation(val_loader, FCCM_net, epoch, train_writer)
        
        is_best = loss <= best_error
        best_error = min(best_error, loss)

        save_checkpoint(
                args.save_path, {
                    'epoch': epoch + 1,
                    'state_dict': FCCM_net.module.state_dict()
                },  {                
                    'epoch': epoch + 1,
                    'state_dict': optimizer.state_dict()
                }, is_best)

    
    
def train(train_loader, FCCM_net, optimizer, epoch_size,  train_writer=None):
    global args, n_iter
    average_loss = 0
    FCCM_net.train()

    for i, (CT_img, ground_truth) in enumerate(tqdm(train_loader)):
        
        CT_img = Variable(CT_img[0].cuda())
        
        ground_truth = torch.tensor((ground_truth.float())).cuda().squeeze(1)
        if args.normalization == 'max':
            CT_img = max_normalize(CT_img)
        if args.normalization =='mean':
            CT_img = mean_normalize(CT_img)

        predict_result = FCCM_net(CT_img)
        #print(predict_result.size())
        #print(ground_truth.size())
        # print(ground_truth[:, -1],predict_result[:,-1])
        classification_loss = nn.functional.binary_cross_entropy(torch.sigmoid(predict_result[:,:-1]), ground_truth[:, :-1])
        regression_loss = ((predict_result[:,-1] - ground_truth[:, -1]) **2).mean()

        loss = classification_loss + regression_loss
        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        average_loss += loss.item()

        if i > 0 and n_iter % args.print_freq == 0:
            train_writer.add_scalar('classification_loss', classification_loss.item(), n_iter)
            train_writer.add_scalar('regression_loss', regression_loss.item(), n_iter)
        if  n_iter % (args.print_freq*40) == 0:
            train_writer.add_image('Input image',
                                    tensor2array(CT_img.data[0].cpu(), max_value=None, colormap='bone'),
                                    n_iter)
        n_iter+=1
    return average_loss/i

import numpy as np
def validation(val_loader, FCCM_net, epoch, train_writer=None):
    global args, n_iter

    FCCM_net.eval()
    ACC_CN = 0
    ACC_H = 0
    ACC_OS = 0
    ACC_CT = 0
    cnt = 0
    ERROR_survive = 0
    for i, (CT_img, ground_truth) in enumerate(tqdm(val_loader)):
        
        CT_img = Variable(CT_img.cuda())
        # print(CT_img.size())
        
        ground_truth = (ground_truth.float())[:,0,:].cpu().detach().numpy()
        if args.normalization == 'max':
            CT_img = max_normalize(CT_img)
        if args.normalization =='mean':
            CT_img = mean_normalize(CT_img)

        predict_result = FCCM_net(CT_img)
        #print(predict_result.size())
        #print(ground_truth.size())
        # print(ground_truth[:, -1],predict_result[:,-1])

        # b = predict_result.size()[0]

        predict_classification = torch.sigmoid(predict_result[:,:-1]).cpu().detach().numpy()
        predict_regression = predict_result[:,-1].cpu().detach().numpy()
        b= predict_classification.shape [0]

        # print(predict_classification.size(),predict_classification[:, 0:4].size())
        Clinical_N_stage = np.argmax(predict_classification[:, 0:4],1)
        Histology = np.argmax(predict_classification[:, 4:9],1)
        Overall_Stage = np.argmax(predict_classification[:, 9:13],1)
        Clinical_T_stage = np.argmax(predict_classification[:, 13:18],1)

        GT_Clinical_N_stage = np.argmax(ground_truth[:, 0:4],1)
        GT_Histology = np.argmax(ground_truth[:, 4:9],1)
        GT_Overall_Stage = np.argmax(ground_truth[:, 9:13],1)
        GT_Clinical_T_stage = np.argmax(ground_truth[:, 13:18],1)

        # print(type(Clinical_N_stage),type(GT_Clinical_N_stage))
        ACC_Clinical_N_stage = (np.where(Clinical_N_stage == GT_Clinical_N_stage, 
                                    np.ones_like(GT_Clinical_N_stage), np.zeros_like(GT_Clinical_N_stage))).sum()/b
        ACC_Histology = (np.where(Histology == GT_Histology, 
                                    np.ones_like(GT_Histology), np.zeros_like(GT_Histology))).sum()/b
        ACC_Overall_Stage = (np.where(Overall_Stage == GT_Overall_Stage, 
                                    np.ones_like(GT_Overall_Stage), np.zeros_like(GT_Overall_Stage))).sum()/b
        ACC_Clinical_T_stage = (np.where(Clinical_T_stage == GT_Clinical_T_stage, 
                                    np.ones_like(GT_Clinical_T_stage), np.zeros_like(GT_Clinical_T_stage))).sum()/b
        cnt += 1
        ACC_CN += ACC_Clinical_N_stage
        ACC_H += ACC_Histology
        ACC_OS += ACC_Overall_Stage
        ACC_CT += ACC_Clinical_T_stage

        ERROR_survive += np.abs((predict_regression-ground_truth[:, -1])/predict_regression).sum()/b 

    train_writer.add_scalar('ACC_Clinical_N_stage', (ACC_CN/cnt), epoch)
    train_writer.add_scalar('ACC_Histology', (ACC_H/cnt), epoch)
    train_writer.add_scalar('ACC_Overall_Stage', ACC_OS/cnt, epoch)
    train_writer.add_scalar('ACC_Clinical_T_stage', ACC_CT/cnt, epoch)
    train_writer.add_scalar('ERROR_survive', ERROR_survive/cnt, epoch)

    return ACC_CN/cnt, ACC_H/cnt, ACC_OS/cnt, ACC_CT/cnt, ERROR_survive/cnt





if __name__ == '__main__':

    main()