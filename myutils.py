__author__ = 'Qi'
# Created by on 12/3/21.

import torch
import os
import shutil
import pandas as pd
#
# def adjust_lr_lambda(args,  epoch, optimizer):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     epoch = epoch + 1
#     if args.epochs == 120:
#         if epoch > 90:
#             lr = args.lr * 0.1
#             args.lamda = 0
#         else:
#             lr = args.lr
#             args.lamda = 0
#     elif args.epochs  == 200:
#         if epoch <= 5:
#             lr = args.lr * epoch / 5
#             args.lamda = 0
#         elif epoch >= 180:
#             lr = args.lr * 0.0001
#             args.lamda = 0
#         elif epoch >= 160:
#             lr = args.lr * 0.01
#             args.lamda = 0
#         else:
#             lr = args.lr
#             args.lamda = 0
#     else:
#         if epoch <= 5:
#             args.lamda = 0
#             lr = args.lr * epoch / 5
#         elif epoch >= 60:
#              lr = args.lr * 0.1
#              args.lamda = 0
#         else:
#              lr = args.lr
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#
#     print(epoch, 'Epoch : ', lr)
#     return lr



def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.float().topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].float().sum()
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class ResultsLog(object):
    def __init__(self, path='results.csv', plot_path=None):
        self.path = path
        self.plot_path = plot_path or (self.path + '.html')
        self.figures = []
        self.results = None

    def add(self, **kwargs):
        # pass
        df = pd.DataFrame([kwargs.values()], columns=kwargs.keys())
        if self.results is None:
            self.results = df
        else:
            self.results = self.results.append(df, ignore_index=True)

    def save(self, title='Training Results'):


        if len(self.figures) > 0:
            if os.path.isfile(self.plot_path):
                os.remove(self.plot_path)
            # output_file(self.plot_path, title=title)
            # plot = column(*self.figures)
            # save(plot)
            # self.figures = []
        self.results.to_csv(self.path, index=False, index_label=False)

    def load(self, path=None):


        path = path or self.path
        if os.path.isfile(path):
            self.results.read_csv(path)

    def show(self):
        pass
        # if len(self.figures) > 0:
        #     plot = column(*self.figures)
        #     show(plot)

    def plot(self, xs, ys, *kargs, **kwargs):
        pass
        # fig = figure(*kargs, **kwargs)
        # x_list_value = self.results[xs].values.T.tolist()
        # y_list_value = self.results[ys].values.T.tolist()
        # fig.multi_line(xs = x_list_value, ys = y_list_value)
        # self.figures.append(fig)

    def image(self, *kargs, **kwargs):
        pass
        # fig = figure()
        # fig.image(*kargs, **kwargs)
        # self.figures.append(fig)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def model_resume(args, model, model_new):
    checkpoint = torch.load(model) #, map_location='cuda:0'
    best_acc1 = checkpoint['best_acc1']
    args.resumed_epoch = checkpoint['best_acc1']
    model_new.module.load_state_dict(checkpoint['state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer'])
    print("=> loaded checkpoint '{}' (epoch {}), acc {}"
          .format(args.resume, args.resumed_epoch, best_acc1))


def save_checkpoint_epoch(state, is_best, path='.',):
    filename = os.path.join(path, '%s-th_epoch_checkpoint.pth.tar' % state['epoch'])
    torch.save(state, filename)
    if is_best:
        shutil.move(filename, 'model_best.pth.tar')

def load_checkpoint_epoch(epoch, path='.'):
    filename = os.path.join(path, '%s-th_epoch_checkpoint.pth.tar' % str(epoch))
    return torch.load(filename)

def load_checkpoint_iter(iter, path='.'):
    filename = os.path.join(path, '%s-th_iter_checkpoint.pth.tar' % str(iter))
    return torch.load(filename)


def load_checkpoint_best(path='.'):
    filename = os.path.join(path, 'model_best.pth.tar')
    return torch.load(filename)