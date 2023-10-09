__author__ = 'Qi'
# Created by on 12/3/21.

__author__ = 'Qi'
# Created by on 1/10/21.
import os

import torch
from torch.utils.data import Dataset, DataLoader

from c_analysis.preprocess import get_transform_medium_scale_data, get_data_transform_ImageNet_iNaturalist18
from mydataset import LT, featLT, get_cls_num_list, get_imbalanced_dataset

RGB_statistics = {
    'iNaturalist18': {
        'mean': [0.466, 0.471, 0.380],
        'std': [0.195, 0.194, 0.192]
    },
    'ImageNet': {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    },
    'ImageNet_LT': {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }
}




def myDataLoader_imagenet(args, data_root, batch_size, phase, sampler=None, num_workers=4, shuffle=True):
    assert phase in {'train', 'val', 'test'}
    if 'LT' in args.dataset:
        key = 'ImageNet_LT'
        txt = f'./data/ImageNet_LT/ImageNet_LT_{phase}.txt'
    else:
        key = 'ImageNet'
        txt = f'./data/ImageNet/ImageNet_{phase}.txt'

    rgb_mean, rgb_std = RGB_statistics[key]['mean'], RGB_statistics[key]['std']


    if phase == 'val' and args.stages == 2:
        transform = get_data_transform_ImageNet_iNaturalist18('train', rgb_mean, rgb_std)
    else:
        transform = get_data_transform_ImageNet_iNaturalist18(phase, rgb_mean, rgb_std)

    set_imagenet = LT(data_root, txt, transform)
    # print(f'===> {phase} data length {len(set_imagenet)}')
    # print('Shuffle is %s.' % shuffle)
    return DataLoader(dataset=set_imagenet, batch_size=batch_size, sampler= sampler, shuffle=shuffle, num_workers=num_workers)


# /dual_data/not_backed_up/imagenet-2012/ilsvrc
def myDataLoader_iNaturalist18(args, data_root, batch_size, phase, sampler=None, num_workers=4, shuffle=True, imb_factor = 0.01):

    assert  phase in {'train', 'val'} , "There is no test phase for iNaturalist18"
    key = 'iNaturalist18'
    txt = f'./data/iNaturalist18/iNaturalist18_{phase}.txt'

    print(f'===> Loading iNaturalist10 data from {txt}')
    rgb_mean, rgb_std = RGB_statistics[key]['mean'], RGB_statistics[key]['std']


    if phase == 'val' and args.stages == 2:
        transform = get_data_transform_ImageNet_iNaturalist18('train', rgb_mean, rgb_std, key = 'Naturalist18')
    else:
        transform = get_data_transform_ImageNet_iNaturalist18(phase, rgb_mean, rgb_std, key = 'Naturalist18')


    set_imagenet = LT(data_root, txt, transform)
    print(f'===> {phase} data length {len(set_imagenet)}')



    return DataLoader(dataset=set_imagenet, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                          sampler = sampler)


def get_train_val_test_feature_loader(args):

    if args.dataset == 'iNaturalist18':
        root = '/data/qiqi/constrainedDRO/feat_iNaturalist18/'
    elif args.dataset == 'imagenet-LT':
        root = '/data/qiqi/constrainedDRO/feat_imagenet-LT/'


    train_datasets = featLT(root, 'train')
    train_loader = DataLoader(dataset=train_datasets, batch_size= args.batch_size, sampler= None, shuffle=True, num_workers=0)
    val_datasets = featLT(root, 'val')
    val_loader = DataLoader(dataset=val_datasets, batch_size= args.batch_size, sampler= None, shuffle=True, num_workers=0)
    test_loader = None

    if args.dataset == 'imagenet-LT':
        test_datasets = featLT(root, 'test')
        test_loader = DataLoader(dataset=test_datasets, batch_size=args.batch_size, sampler=None, shuffle=True,
                            num_workers=0)

    return train_loader, val_loader, test_loader




def get_train_val_test_loader(args, train_sampler = None):
    sampler = train_sampler
    test_loader = None

    if args.dataset == 'imagenet-LT':
        if 'argon' in os.uname()[1]:
            args.data_root ="/nfsscratch/qqi7/imagenet/"
        elif 'amax' in os.uname()[1]: # 210.28.134.11
            args.data_root = "/data/imagenet/imagenet/"
        elif 'test-X11DPG-OT' in os.uname()[1]:
            args.data_root = "/home/qiuzh/imagenet/"
        else:
            args.data_root = '/dual_data/not_backed_up/imagenet-2012/ilsvrc/'


        if sampler is not None:
            train_loader = myDataLoader_imagenet(args, args.data_root, args.batch_size, 'train', sampler=sampler,
                                                 num_workers=args.workers,  shuffle=False)
        else:
            train_loader = myDataLoader_imagenet(args, args.data_root, args.batch_size, 'train',
                                                 num_workers=args.workers, shuffle=True)

        val_loader =  myDataLoader_imagenet(args, args.data_root, args.batch_size if args.batch_size != 1 else 64, 'val',  num_workers = args.workers, shuffle = False)
        test_loader =  myDataLoader_imagenet(args, args.data_root, args.batch_size if args.batch_size != 1 else 64, 'test', sampler = sampler, num_workers = args.workers, shuffle = False)
        args.cls_num_list = get_cls_num_list(args)

    elif args.dataset == 'imagenet':
        if 'argon' in os.uname()[1]:
            args.data_root ="/nfsscratch/qqi7/imagenet/"
        elif 'amax' in os.uname()[1]: # 210.28.134.11
            args.data_root = "/data/imagenet/imagenet/"
        elif 'test-X11DPG-OT' in os.uname()[1]:
            args.data_root = "/home/qiuzh/imagenet/"
        else:
            args.data_root = '/dual_data/not_backed_up/imagenet-2012/ilsvrc/'

        if sampler is not None:
            train_loader = myDataLoader_imagenet(args, args.data_root, args.batch_size, 'train', sampler=sampler,
                                                 num_workers=args.workers, shuffle=False)
        else:
            train_loader = myDataLoader_imagenet(args, args.data_root, args.batch_size, 'train',
                                          num_workers=args.workers, shuffle=True)
        val_loader = myDataLoader_imagenet(args, args.data_root, args.batch_size // 4, 'val',
                                        num_workers=args.workers, shuffle=False)
        test_loader = myDataLoader_imagenet(args, args.data_root, args.batch_size // 4, 'test',
                                         num_workers=args.workers, shuffle=False)
        args.cls_num_list = get_cls_num_list(args)

    elif args.dataset == 'iNaturalist18':
        if 'argon' in os.uname()[1]:
            args.data_root = "/nfsscratch/qqi7/iNaturalist2018/"
        elif 'amax' in os.uname()[1]: # 210.28.134.11
            args.data_root = "/data/iNaturalist2018/"
        else:
            args.data_root = "/dual_data/not_backed_up/iNaturalist2018/"

        if sampler is not None:
            train_loader = myDataLoader_iNaturalist18(args, args.data_root, args.batch_size, 'train', sampler = sampler,
                                           num_workers=args.workers, shuffle=False)
        else:
            train_loader = myDataLoader_iNaturalist18(args, args.data_root, args.batch_size, 'train',
                                                      num_workers=args.workers, shuffle=True)



        val_loader = myDataLoader_iNaturalist18(args, args.data_root, 256, 'val',
                                         num_workers=args.workers, shuffle=False)

        args.cls_num_list = get_cls_num_list(args)

    else:
        default_transform = {
            'train': get_transform_medium_scale_data(args.dataset,
                                   input_size=args.input_size, isTrain=True),
            'eval': get_transform_medium_scale_data(args.dataset,
                                  input_size=args.input_size, isTrain=False)
        }

        if args.dataset == 'C2':
            traindir = './data/C2/im_train_' + str(args.im_ratio) + '/'
            valdir = './data/C2/val/'
            torch.manual_seed(777)
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            if sampler is not None:

                train_loader = torch.utils.data.DataLoader(
                    datasets.ImageFolder(traindir,
                                         transforms.Compose([
                                             # transforms.Lambda(shear),
                                             transforms.RandomResizedCrop(224),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             normalize,
                                         ])),
                    batch_size=args.batch_size,
                    sampler=sampler,
                    shuffle=False,
                    num_workers=8,
                    pin_memory=True)
            else:
                 train_loader = torch.utils.data.DataLoader(
                    datasets.ImageFolder(traindir,
                                            transforms.Compose([
                                                # transforms.Lambda(shear),
                                                transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                normalize,
                                            ])),
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=8,
                    pin_memory=True)

            val_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(valdir,
                                     transforms.Compose([
                                         transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         normalize,
                                     ])),
                batch_size=args.batch_size // 2,
                shuffle=False,
                num_workers=8,
                pin_memory=True)

        elif args.dataset == 'melanoma':
            traindir = './data/C2/im_train_' + str(args.im_ratio) + '/'
            valdir = './data/C2/val/'
            #torch.manual_seed(777)
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            if sampler is not None:
                train_loader = torch.utils.data.DataLoader(
                    datasets.ImageFolder(traindir,
                                         transforms.Compose([
                                             # transforms.Lambda(shear),
                                             transforms.RandomResizedCrop(224),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             normalize,
                                         ])),
                    batch_size=args.batch_size,
                    shuffle=False,
                    sampler = sampler,
                    num_workers=8,
                    pin_memory=True)
            else:
                train_loader = torch.utils.data.DataLoader(
                    datasets.ImageFolder(traindir,
                                         transforms.Compose([
                                             # transforms.Lambda(shear),
                                             transforms.RandomResizedCrop(224),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             normalize,
                                         ])),
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=8,
                    pin_memory=True)

            val_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(valdir,
                                     transforms.Compose([
                                         transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         normalize,
                                     ])),
                batch_size=args.batch_size // 2,
                shuffle=False,
                num_workers=8,
                pin_memory=True)
        else:



            transform = default_transform

            train_data, val_data = get_imbalanced_dataset(args.dataset, args.im_ratio, transform)
            # print(train_data)
            # print(val_data)
            # print(transform)
            # print(args.im_ratio)

            # print(train_data)
            # torch.manual_seed(777)

            # self.dataset = dataset
            # self.indices = indices
            # print(">>>>>>>>>>>>>", train_data.dataset, train_data.indices, "<<<<<<<<<<<<<<<<<<<<")
            # print(">>>>>> train_datasets <<<<<<<:", train_data.dataset.data)


            # print(">>>>:", train_data, type(train_data))
            # sorted_indices = sorted(train_data.indices)
            # dict_sorted_indices= {i:False for i in range(len(train_data.dataset))}
            # train_data_data = []
            # train_labels = []
            # for i in sorted_indices:
            #     dict_sorted_indices[i] = True
            #
            # for i, (data, label) in enumerate(train_data.dataset):
            #     if dict_sorted_indices[i]:
            #         train_data_data.append(data)
            #         train_labels.append(torch.tensor(label))

            # train_data = torch.stack(train_data_data)
            # train_labels = torch.stack(train_labels)
            # #
            # print(">>>>>>>>>:", train_data.size(), train_labels)
            # train_data = indexCIFARDatasets(train_data, train_labels, default_transform['train'])
            # val_data = indexCIFARDatasets(val_data.data, val_data.targets, default_transform['eval'])

            # val_data_data = []
            # val_labels = []
            # for data in val_data.data:
            #     val_data_data.append(torch.tensor(np.transpose(data, (2, 0, 1))))
            # for label in val_data.targets:
            #     val_labels.append(torch.tensor(label))
            #
            # val_data = torch.stack(val_data_data)
            # val_labels = torch.stack(val_labels)
            # print(val_data.size(), val_labels.size())

            if sampler is not None:
                train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, sampler=sampler,
                                                           shuffle=False,
                                                           num_workers=args.workers, pin_memory=True)
            else:
                train_loader = torch.utils.data.DataLoader(train_data, batch_size= args.batch_size,  shuffle=True,
                                                       num_workers=args.workers, pin_memory=True)

            val_loader = torch.utils.data.DataLoader(val_data, batch_size=100, shuffle=False,
                                                     num_workers=args.workers, pin_memory=True)
            print(len(train_loader), len(val_loader))
    return train_loader, val_loader, test_loader



