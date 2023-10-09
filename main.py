__author__ = 'Qi'
# Created by on 12/3/21.
import argparse
import os
import time
from datetime import datetime
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from mydataset import get_imbalanced_dataset, get_num_classes
import models
from myutils import ResultsLog, model_resume
# from preprocess import get_transform_medium_scale_data
from balgs import SPD, FastDRO, Cavr_Chisaqure_Baseline, daul_SGM, PG_SMD2 #Cavr_Chisaqure_Baseline
from qalgs import RECOVER, SCCMA, ACCSCCMA, myABSGD
import numpy as np


parser = argparse.ArgumentParser(description="Pytorch PLCOVER Training")
parser.add_argument('--results_dir', metavar="RESULTS_DIR", default='./TrainingResults', help = 'results dir')

parser.add_argument('--saveFolder', metavar = 'SAVE',  default='',help='save folder')
parser.add_argument('--res_filename', default='', type = str, help = 'results file name')
parser.add_argument('--dataset',  metavar='DATASET', default='cifar10',
                    help = 'dataset name or folder')

parser.add_argument('--model', metavar = 'MODEL', default='resnet', help ='model architecture')
parser.add_argument('--type', default='torch.cuda.FloatTensor',
                    help = 'types of tensor - e.g torch.cuda.FloatTensor')
parser.add_argument('--gpus',  default='0', help = 'gpus used for training - e.g 0,1,2,3')
parser.add_argument('--workers', default='8', type = int, metavar='N',
                    help='number of data loading workers (default:256)')
parser.add_argument('--batch-size', default=256, type=int, metavar='N',
                    help = 'mini-batch size (default:256)')
parser.add_argument('--optimizer', default='SGD',type=str, metavar='OPT',
                    help='optimizer function used')
parser.add_argument('--momentum', default=0, type = float, metavar="M",
                    help = "momentum parameter of SHB or SNAG")
parser.add_argument('--scale_size', default=32, type=int, help = 'image scale size for data preprocessing')
parser.add_argument('--input_size', default=32, type=int, help = 'the size of image. e.g. 32 for cifar10, 224 for imagenet')
parser.add_argument('--works', default=8, type=int, help = 'number of threads used for loading data')
parser.add_argument('--weight_decay', default=2e-4, type=float, help ='weight decay parameters')
parser.add_argument('--print_freq', '-p', default=50, type = int,
                    help = 'print frequency (default:50)')
parser.add_argument('--mvg_g_obj', default= 1, type = float, help ='initialized g objective')
# number of restart batches: restart_init_loop * batchsize
parser.add_argument('--restart_init_loop', default=5, type = int,
                    help = 'restart minibatch size = restart_init_loop * batchsize')
parser.add_argument('--start_training_time', type = float, help = 'Overall training start time')
parser.add_argument('--lamda', default=5, type = float, help = 'parameters of regularization')
parser.add_argument('--lamda1', default=5, type = float, help = 'initial lambd1 for the constraints such that lambda >= lambda1')
parser.add_argument('--lamda0', default=1e-3, type = float, help = 'lambda0 to make the DRO objective smooth')
parser.add_argument('--beta', default=0.1, type = float, help = 'momentum parameters for SCCMA')
parser.add_argument('--class_tau', default=0, type = float, help = 'class level dro')
parser.add_argument('--frozen_aside_fc', default=False, type=eval, choices=[True, False],
                    help='whether frozen the feature layers (First three block)')
parser.add_argument('--is_train_last_block', default=False, type=eval, choices=[True, False],
                    help='whether frozen the feature layers (First three block)')
parser.add_argument('--frozen_aside_linear', default=False, type=eval, choices=[True, False], help = 'For frozen resnet20 last layers')
parser.add_argument('--pretrained', default=False, type=eval, choices=[True, False],
                    help='Wether use pretrained model')
# boolean variable
parser.add_argument('--nesterov', default=False, type=eval, choices=[True, False],
                    help = 'This is used to determine whether we use SNAG')
parser.add_argument('--resume', default=False, type=eval, choices=[True, False],
                    help = 'Training from scratch (False) or from the saved check point')
###Tuning Parameters
parser.add_argument('--epochs', default=0, type=int,
                    help = 'number of total epochs')
parser.add_argument('--lr', default=0.1, type=float, metavar='WLR',
                    help='initial learning rate of w')
parser.add_argument('--plr', default=0.005, type = float, help = 'Dual Variable P')
parser.add_argument('--rho', default=1e-4, type = float, help = 'Constraint of DRO: rho')

# Loading Models Parameters
parser.add_argument('--resumed_epoch', default=0, type=int, help = "continuing training from a save check point")
parser.add_argument('--stages', default='1，2，3，4', type = str, help = 'start epochs of each stages')
parser.add_argument('--start_epochs', default=0, type=int, help = "start training epochs: default 0 in common training and start from loaded_epochs - 1 after loading the check point ")
parser.add_argument('--ith_init_run', default=0, type=int, help = "ith-initial weights")
parser.add_argument('--num_classes', default=10, type=int, help = "classes of different datasets")
parser.add_argument('--im_ratio', default=0.2, type=float, help = "imbalance ratio of datasets")
parser.add_argument('--DR', default=10, type=int, help = 'Decay Rate of Different Stages')
parser.add_argument('--binary', default=False, type=eval, choices=[True, False], help = 'Whether perform binary classification.')
parser.add_argument('--auc', default=False, type = eval, choices=[True, False], help = 'calculating AUC in binary classification')
parser.add_argument('--curlr', default=0.1, type=float,
                    help='current learning rate')
parser.add_argument('--lrlambda', default= 0.1, type=float,
                    help='current lambda rate')
parser.add_argument('--curbeta', default=0.1, type=float,
                    help='current learning rate')
parser.add_argument('--obj', default='ERM', type=str,
                    help='optimization objective of the loss')
parser.add_argument('--alg', default='PDSGD', type = str, choices=['ABSGD', 'PG_SMD2', 'RECOVER', 'FastDRO', 'PDSGD', 'ACCSCCMA', 'SCCMA', 'ROBSGD', 'RECOVER', 'MBSGD', 'CAVRCHISQUARE', 'dual_SGM'], help = 'The choice of algorithms')
parser.add_argument('--stablization', default=False, type = eval, choices=[True, False], help = 'whether using stablization for SCDRO(SCCMA) ')

# Constrained DRO
parser.add_argument('--sampleType', default='uniform', type=str, help = 'Sampling methods')
parser.add_argument('--random_seed', default=0, type=int, help='independent random seed')
parser.add_argument('--a_t', default=0.9, type = float, help = 'moving average parameter of recover')
parser.add_argument('--y_t', default=0, type = float, help = 'stochastic estimator of inner exp objective')

parser.add_argument('--size', type=float, default=0.1)
parser.add_argument('--reg', type=float, default=0.01)
parser.add_argument('--geometry', type=str, default='cvar',
                    choices=['cvar', 'chi-square'])
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--mylambda', type = float , default=5, help = 'Tempurature parameter for absgd')




def main():

    global args, best_prec1
    best_prec1 = 0
    args = parser.parse_args()
    args.start_training_time = time.time()

    if args.saveFolder is '':
        args.saveFolder = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    args.results_dir = os.path.join(args.results_dir, args.saveFolder) # root_dir + save Folder
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    results_file = os.path.join(args.results_dir, args.res_filename + '_results.csv')
    results = ResultsLog(results_file)


    if 'cuda' in args.type:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        args.gpus = [int(i) for i in args.gpus.split(',')]
        cudnn.benchmark = True
    else:
        args.gpus = None


    args.num_classes = get_num_classes(args)
    model = models.__dict__[args.model]
    model_cur = model(pretrained = args.pretrained, num_classes = args.num_classes, data = args.dataset)
    model_prev = model(pretrained = args.pretrained, num_classes = args.num_classes, data = args.dataset)


    print("length of model:", len(model_cur.state_dict().keys()))
    if args.frozen_aside_fc or args.frozen_aside_linear:
        print("We are just training part of the neural network")
        network_frozen(args, model_cur)
        network_frozen(args, model_prev)

    if args.gpus and len(args.gpus) >= 1:
        print("We are running the model in GPU :", args.gpus)

        model_cur = torch.nn.DataParallel(model_cur)
        model_prev = torch.nn.DataParallel(model_prev)
        model_cur.type(args.type)
        model_prev.type(args.type)

    if args.resume:
        print("We are loading from a pretrained ce model.")
        if os.path.isfile(args.resume):
            model_resume(args, args.resume, model_cur)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    if args.dataset == 'cifar10':
        args.im_ratio = args.im_ratio
    elif args.dataset == 'cifar100':
        args.im_ratio = args.im_ratio


    if args.alg == "SPD":
        '''
        For the convex, PDSGD == SPD;
        '''
        print(args)
        SPD(args, model_cur, results)
    elif args.alg == 'PG_SMD2':
        '''
        For the convex, PDSGD == SPD;
        '''
        print(args)
        PG_SMD2(args, model_cur, results)
    elif args.alg == 'SCCMA':
        print(args.lamda1, args.rho)
        SCCMA(args, model_cur, results)
    elif args.alg == 'RECOVER':
        print('We are optimizing the  model using {}'.format(args.alg))
        # print(args)
        RECOVER(args, model_cur, results)
    elif args.alg == 'FastDRO':
        print(args)
        FastDRO(args, model_cur, results)
    elif args.alg == 'ACCSCCMA':
        print(args)
        ACCSCCMA(args, model_cur, model_prev, results)
    elif args.alg == 'CAVRCHISQUARE':
        print(args.geometry, args.size, args.reg, args.alg)
        Cavr_Chisaqure_Baseline(args, model_cur, results)
    elif args.alg == 'dual_SGM':
        print(args)
        daul_SGM(args, model_cur, results)
    elif args.alg == 'ABSGD':
        print(args)
        myABSGD(args, model_cur, results)
    else:
        pass





def network_frozen(args, model):
    last_block_number = 0
    if args.model == "resnet152":
        last_block_number = 2
    elif args.model == 'resnet50':
        last_block_number = 2
    elif args.model == 'resnet10':
        last_block_number = 0

    last_block_pattern = 'layer4.' + str(last_block_number)

    # last_block_pattern = 'layer4.'
    if args.model == 'resnet32':
        last_block_pattern = 'layer3.4'


    total_layers = 0
    for param_name, param in model.named_parameters():  # (self.networks[key]):  # frozen the first 3 block
        total_layers +=1
        if 'fc' not in param_name and "linear" not in param_name:
            param.requires_grad = False
            if args.is_train_last_block:
                if last_block_pattern in param_name:
                    param.requires_grad = True

    cnt_layers = 0
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            cnt_layers += 1
            # print(param_name)
    print("{0}/{1} number of trained layers".format(cnt_layers, total_layers))





if __name__ == '__main__':
    main()


