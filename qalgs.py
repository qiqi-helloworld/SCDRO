__author__ = 'Qi'
# Created by on 12/18/21.
import torch, copy, time
import numpy as np
from myutils import AverageMeter, accuracy, save_checkpoint_epoch
from myDataLoader import  get_train_val_test_loader, get_train_val_test_feature_loader
import torch.nn as nn
import torch.nn.functional as F
from sklearn import  metrics
from torch.autograd import Variable
from torch.nn.parallel._functions import Broadcast
import torch.cuda.comm as comm
from torch.autograd import Function
from balgs import validate, adjust_curlr_beta, validate_cifar_val_loader
import wandb
import time
from absgd.losses import ABLoss
from absgd.optimizers import ABSGD, ABAdam
import torch.optim as optim
from torch.optim.lr_scheduler  import MultiStepLR

def update_states(self, **kwargs):
    for key, value in kwargs.iteritems():
        setattr(self, key, value)

class ReduceAddCoalesced(Function):
    @staticmethod
    def forward(ctx, destination1, destination2, num_inputs, *grads):
        ctx.target_gpus = [grads[i].get_device() for i in range(0, len(grads), num_inputs)]

        # print("len(grads): ", len(grads))
        grads_1 = [grads[i:i + num_inputs]
                   for i in range(0, len(grads) // 2, num_inputs)]
        grads_2 = [grads[i:i + num_inputs]
                  for i in range(len(grads) // 2, len(grads), num_inputs)]
        global LASTHALFGRADIENT
        # \nabla f(w_t, \xi_{t+1}), saved for next iteration
        LASTHALFGRADIENT = comm.reduce_add_coalesced(grads_2, destination2)
        return comm.reduce_add_coalesced(grads_1, destination1) # \nabla f(w_t, \xi_{t}), return back to GPU 0 to update w_t





def RECOVER(args, model_new, results):

    '''
    > Hyperparameter lambda is controled by lambda1
    > The initial lambda is set 100
    > Then we decay lambda at stagewise learning rate decay epoch.
    > a_t denotes the momentum parameter of  algorithm
    # @ args.cur_lambda
    # @ args.a_t
    # @ args.\rho # constraint

    '''


    wandb.init(config=args, project="SCCMA", entity="qiqi-helloworld")

    #print(args.gpus)
    GPULEN = len(args.gpus)

    if args.model == 'resnet50':
        total_gradient_layer = 161
    elif args.model == 'resnet32':
        total_gradient_layer = 95
    elif args.model == 'resnet20':
        total_gradient_layer = 59

    def backward(ctx, *grad_outputs):
        # print("Hello: using self defined backword.")
        # print('grad_out_outputs:', len(grad_outputs))
        return (None,) + ReduceAddCoalesced.apply(0, GPULEN // 2, total_gradient_layer, *grad_outputs)

    # resnet20 59,resnet32: 95 resnet50:161
    ivd_criterion = nn.CrossEntropyLoss(reduction='none')
    CE_criterion = nn.CrossEntropyLoss()
    SGDoptimizer = optim.SGD(model_new.parameters(), lr=args.lr, momentum=0.9)

    train_loader, val_loader, test_loader = get_train_val_test_loader(args, None)

    start_time = time.time()
    args.curlamda = 100
    if args.epochs == 200:
        args.stages = [0, 160, 180]
    elif args.epochs == 120:
        args.stages = [0, 90]
    elif args.epochs == 60:
        args.stages = [0, 30]

    Broadcast.backward = backward
    global inputs_1, targets_1, per_epoch_time, state
    state = dict()
    inputs_1, targets_1 = None, None

    best_acc1 = 0
    for epoch in range(args.resumed_epoch, args.epochs):

        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        adjust_curlr_beta(epoch, args, optimizer=None)
        model_new.train()

        # if epoch < args.stages[0]:
        #     print('{}-th epochs of learning features using SGD, network {}'.format(epoch, args.model))
        #     for batch_idx, (_, inputs, targets) in enumerate(train_loader):
        #         # print(inputs, targets)
        #         inputs, targets = inputs.cuda(), targets.cuda()
        #         outputs, _ = model_new(inputs)
        #         ce_loss = CE_criterion(outputs, targets)
        #
        #         SGDoptimizer.zero_grad()
        #         ce_loss.backward()
        #         SGDoptimizer.step()
        #         acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        #         losses.update(ce_loss.item(), inputs.size(0))
        #         top1.update(acc1.item(), inputs.size(0))
        #         top5.update(acc5.item(), inputs.size(0))
        #         # print('loss:', losses.avg)
        #
        # else:
        #     if epoch == args.stages[0]:
        if epoch == args.stages[1]:
            args.curlamda = args.lamda1

        train_iter = enumerate(train_loader)
        done_looping = False
        batch_idx = 0
            # initializing for parallel training
        if inputs_1 is None and targets_1 is None:
            _, (_, inputs_1, targets_1) = next(train_iter)
            batch_idx += 1
        _,(_, inputs_2, targets_2) = next(train_iter)
        batch_idx += 1

        while not done_looping:

            inputs = torch.cat([inputs_1, inputs_2])
            targets = torch.cat([targets_1, targets_2])

            inputs, targets = inputs.cuda(), targets.cuda()
            # print(inputs)
            outputs, _ = model_new(inputs)

            loss1 = ivd_criterion(outputs[0:args.batch_size], targets[0:args.batch_size])
            loss2 = ivd_criterion(outputs[args.batch_size:], targets[args.batch_size:])

            loss1_max = torch.Tensor.detach(torch.max(loss1))/args.curlamda
            loss2_max = torch.Tensor.detach(torch.max(loss2))/args.curlamda

            exp_loss_1 = torch.mean(torch.exp(loss1 / args.curlamda - loss1_max))
            exp_loss_2 = torch.mean(torch.exp(loss2 / args.curlamda - loss2_max))

            exp_loss = (exp_loss_1 + exp_loss_2) / 2
            loss = loss1_max * args.curlamda + args.curlamda * torch.log(
                    torch.mean(torch.exp(loss1 / args.curlamda - loss1_max)))

            model_new.zero_grad()
            exp_loss.backward()

            args.y_t = exp_loss_1 + (1 - args.a_t) * args.y_t # already scaled in previous iteration
            for name, param in model_new.named_parameters():  # load the name and value of every layer.
                if param.requires_grad:
                    if name not in state.keys() or epoch == args.resumed_epoch:
                        state[name] = torch.tensor(0.0) # 1.0 --> 0.1 -->0
                    state[name] = param.grad + (1 - args.a_t) * state[name]
                    param.data.add_(-args.curlr, args.curlamda * state[name] / args.y_t + args.weight_decay * param.data)

            # Update State and Y_t
            # u_t - \exp(frac{\ell(\w_t,\z_{t+1})}{\lambda})
            args.y_t = args.y_t *torch.exp(loss1_max-loss2_max) - exp_loss_2.item()
            # 2. no normalization
            # args.y_t = args.y_t - exp_loss_2.item()

            # v_t - \exp(\frac{\ell(\w_t,\z_{t+1})}{\lambda})\nabla \ell(\w_t,\z_{t+1})
            i = 0
            for name, param in model_new.named_parameters():
                if param.requires_grad:
                    state[name] = state[name] * torch.exp(loss1_max-loss2_max) -  LASTHALFGRADIENT[i].to(0)
                    # 2. no normalization
                    # state[name] = state[name] -  LASTHALFGRADIENT[i].to(0)
                i += 1

            if batch_idx <= len(train_loader) - 1:
                _, (_, inputs_1, targets_1) = next(train_iter)
                inputs_1, inputs_2 = inputs_2, inputs_1
                targets_1, targets_2 = targets_2, targets_1
                batch_idx += 1
            else:
                inputs_1 = inputs_2
                targets_1 = targets_2
                done_looping = True


            if batch_idx % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Train Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                            'Train Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, batch_idx, len(train_loader),
                    loss=losses, top1=top1, top5=top5))



            acc1, acc5 = accuracy(outputs[0:args.batch_size], targets[0:args.batch_size], topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))
            #print('loss:', losses.avg)
        wandb.log({"u_t": args.y_t.item()}, step=epoch)

        if args.epochs > 10:


            if 'cifar' in args.dataset:
                train_loss, train_prec1, train_prec5, train_auc_score = validate_cifar_val_loader(args, train_loader, model_new,
                                                                                CE_criterion,
                                                                                epoch)
                val_loss, val_prec1, val_prec5, val_auc_score = validate_cifar_val_loader(args, val_loader, model_new,
                                                                                          CE_criterion, epoch)
            else:
                train_loss, train_prec1, train_prec5, train_auc_score = validate(args, train_loader,
                                                                                                 model_new,
                                                                                                 CE_criterion,
                                                                                                 epoch)
                val_loss, val_prec1, val_prec5, val_auc_score = validate(args, val_loader, model_new, CE_criterion,
                                                                         epoch)

            if test_loader is not None:
                test_loss, test_prec1, test_prec5, test_auc_score = validate(args, test_loader, model_new, CE_criterion,
                                                                             epoch)

        overall_running_time = (time.time() - start_time) // 60
        is_best = True if val_prec1 >= best_acc1 else False
        best_acc1 = max(best_acc1, val_prec1)
        results.add(epoch=epoch, val_loss=val_loss,
                    train_prec1=train_prec1, val_prec1=val_prec1 if test_loader is None else test_prec1,
                    train_prec5=train_prec5, val_prec5=val_prec5 if test_loader is None else test_prec5,
                    overall_running_time=overall_running_time)
        results.save()


        wandb.log({"lr": args.curlr, 'current lambda': args.curlamda}, step=epoch)
        wandb.log({"train loss": train_loss, 'train acc1': train_prec1, 'train acc5': train_prec5}, step=epoch)
        wandb.log({"test loss": val_loss, 'test acc1': val_prec1 if test_loader is None else test_prec1, 'test acc5': val_prec5 if test_loader is None else test_prec5}, step=epoch)
        wandb.log({"best test acc": best_acc1, 'beta': args.curbeta}, step=epoch)
        # if (epoch + 1) % 10 == 0 or epoch == 0:
        #     save_checkpoint_epoch({
        #         'epoch': args.epochs,
        #         'model': args.model,
        #         'state_dict': model_new.state_dict(),
        #         'best_acc1': best_acc1,
        #     }, is_best, path=args.results_dir)


def SCCMA(args, model_new, results):

    wandb.init(config=args, project="SCCMA", entity="qiqi-helloworld")
    global w_grad_state
    w_grad_state = dict()

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    print(">>>>>>> Random seed:", args.random_seed)

    train_loader, val_loader, test_loader = get_train_val_test_loader(args, None)
    if (args.frozen_aside_fc or args.frozen_aside_linear) and not args.is_train_last_block:
        train_loader, val_loader, test_loader = get_train_val_test_feature_loader(args)


    ivd_criterion = nn.CrossEntropyLoss(reduction='none')
    CE_criterion = nn.CrossEntropyLoss()


    best_acc1 = 0
    args.lamda = args.lamda1
    train_loss, train_prec1, train_prec5, test_prec1, test_prec5 = 0, 0, 0, 0, 0
    mvg_g_obj, mvg_grad_lambda, ivd_loss = args.mvg_g_obj, 0, 0
    # if test_loader is not None:
    #     test_loss, test_prec1, test_prec5, test_auc_score = validate(args, test_loader, model_new, CE_criterion, 0)
    #     print('>>>>>>>> Pretrained test_prec1 {:.3f}'.format(test_prec1))

    print('>>>>>>:', args.stablization)
    max_ivd_loss_prev = -1 # Designed for stablization.
    adjust_beta = 0
    for epoch in range(args.resumed_epoch, args.epochs):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        if args.epochs == 150 and epoch == 90:
            args.lrlambda =  args.lrlambda * 10
        if args.epochs == 30 and epoch == 20:
            # args.lrlambda = args.lrlambda * 10
            pass
            # else: keep args.lrlamda constant

        adjust_curlr_beta(epoch, args)

        if epoch > 160:
            args.curlr = args.curlr * 5
        model_new.train()
        # print("cccc")

        for batch_idx, (_, inputs, targets) in enumerate(train_loader):
            start_time = time.time()
            inputs, targets= inputs.cuda(), targets.cuda()
            outputs, _ = model_new(inputs)

            ivd_loss = ivd_criterion(outputs, targets)
            # old version:
            max_ivd_loss = torch.tensor(max(ivd_loss).item())
            # new version:
            # max_ivd_loss = torch.tensor(max(ivd_loss).item())/args.lamda
            # print('max:', max_ivd_loss)
            if max_ivd_loss_prev < 0:
                max_ivd_loss_prev = max_ivd_loss # or could be a larger number for initialization.

            # history largest loss, to avoid not a number when updates
            # max_ivd_loss_cur = max_ivd_loss
            max_ivd_loss_cur = max(max_ivd_loss_prev, max_ivd_loss)
            # inner objective
            g_obj = torch.exp(ivd_loss/args.lamda - max_ivd_loss_cur)
            # c_g_obj = torch.exp(ivd_loss/args.lamda - max_ivd_loss_cur/args.lamda).detach()
            # print(g_obj.size(), ">>>>: Wrong: ", torch.sum(g_obj).item(), torch.mean(g_obj).item(), 'Correct:', torch.sum(c_g_obj).item(), torch.mean(c_g_obj).item(),": <<<<")
            # inner objective estimator
            mvg_g_obj = (1-args.curbeta)* mvg_g_obj * torch.exp(max_ivd_loss_prev - max_ivd_loss_cur) + args.curbeta * g_obj.detach().mean() # ;  s_t
            # adjust_beta = (1-args.curbeta)*torch.exp(max_ivd_loss_prev - max_ivd_loss_cur)
            # print(g_obj.detach().mean())
            # u_t, gradient of f(g) = \lambda \log (g)
            grad_f = args.lamda/mvg_g_obj
            # chain rule to have the stochastic gradient in terms of \w : \lambda \log (g(w))
            grad_f_g_obj = torch.mean(grad_f*g_obj)
            model_new.zero_grad()
            grad_f_g_obj.backward()
            # updates W
            for name, param in model_new.named_parameters():  # load the name and value of every layer.
                if name not in w_grad_state.keys() and param.requires_grad:
                    w_grad_state[name] = param.grad
                else:
                    if param.requires_grad:
                        # stochastic gradient moving average estimator
                        w_grad_state[name] = (1-args.curbeta)* w_grad_state[name]  +  args.curbeta *  param.grad
                        # updates \w
                        param.data.add_(-args.curlr, w_grad_state[name] + args.weight_decay * param.data)  # for model w, we add weight decay
            # updates \lambda
            grad_g_lambda = - g_obj.detach() * ivd_loss.detach() / (args.lamda**2)

            grad_lambda = grad_f * grad_g_lambda + torch.log(mvg_g_obj) + args.rho + max_ivd_loss_cur
            mvg_grad_lambda = (1 - args.curbeta) * mvg_grad_lambda  +  args.curbeta * grad_lambda.mean()
            args.lamda = args.lamda - args.lrlambda * (mvg_grad_lambda  + args.weight_decay * args.lamda)
            # print('Current lambda {}, Gradient of lambda {}, rho {}, moving average of inner objectve {}'.format(args.lamda, grad_lambda, args.rho, mvg_g_obj))
            # print('max individual loss {}'.format(max_ivd_loss_cur))


            if args.lamda < args.lamda0:
                args.lamda = args.lamda0
                max_ivd_loss_prev = max_ivd_loss_cur


            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            loss = CE_criterion(outputs, targets)
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))
            # print(top1.avg)

            if batch_idx % args.print_freq == 0:
                if args.epochs <= 10:
                    train_loss, train_prec1, train_prc5, train_auc_score = validate(args, train_loader, model_new, CE_criterion, epoch)
                    val_loss, val_prec1, val_prec5, val_auc_score = validate_cifar_val_loader(args, val_loader, model_new, CE_criterion, epoch)
                    print('iter acc1', epoch * len(train_loader) + batch_idx, 4*len(train_loader), train_prec1, val_prec1)
                    # wandb.log({'iter acc1': train_prec1, 'iter val acc1': val_prec1}, step= (epoch * len(train_loader) + batch_idx))
                else:
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Train Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Train Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, batch_idx, len(train_loader), loss=losses, top1=top1, top5=top5))
            # print('Avarage seconds per iteration is {}'.format(time.time() - start_time))

        if args.epochs > 10:
            train_loss, train_prec1, train_prc5, train_auc_score = validate_cifar_val_loader(args, train_loader, model_new, CE_criterion, epoch)
            if 'cifar' in args.dataset:
                val_loss, val_prec1, val_prec5, val_auc_score = validate_cifar_val_loader(args, val_loader, model_new, CE_criterion, epoch)
            else:
                val_loss, val_prec1, val_prec5, val_auc_score = validate_cifar_val_loader(args, val_loader, model_new, CE_criterion, epoch)

            if test_loader is not None:
                test_loss, test_prec1, test_prec5, test_auc_score = validate_cifar_val_loader(args, test_loader, model_new, CE_criterion, epoch)

            overall_running_time = (time.time() - start_time)//60
            best_acc1 = max(best_acc1, val_prec1) if test_loader is None else max(best_acc1, test_prec1)
            results.add(epoch=epoch, val_loss=val_loss,
                        train_prec1 = train_prec1, val_prec1 = val_prec1,test_prec1 = test_prec1 if test_loader is not None else val_prec1,
                        train_prec5 = train_prec5, val_prec5 = val_prec5, test_prec5 = test_prec5 if test_loader is not None else val_prec5,
                        overall_running_time=overall_running_time, adjust_beta = adjust_beta)
            results.save()

            ##### Print on the screen.
            #if epoch % 15 == 0:
            output = ('Train: [{0}/{1}], lr: {lr:.5f}\t'
                          'Train Loss {train_loss:.4f} Val Loss {val_loss:.4f}\t'
                          'Train Prec@1 {train_prec1:.3f} Val Prec@1 {val_prec1:.3f} Test Prec@1 {test_prec1:.3f} \t'
                          'Train Prec@5 {train_prec5:.3f} Val Prec@5 {val_prec5:.3f} Test Prec@5 {test_prec5:.3f}, Adjust_beta {adjust_beta:.3f}'.format(
                    epoch, args.epochs, train_loss=train_loss, val_loss=val_loss,
                    train_prec1=train_prec1, val_prec1=val_prec1, test_prec1 = test_prec1 if test_loader is not None else val_prec1,
                train_prec5=train_prec5, val_prec5=val_prec5, test_prec5 = test_prec5 if test_loader is not None else val_prec5, lr = args.curlr, adjust_beta = adjust_beta))
            print(output)
            print("Lambda Variable value: ", str(args.lamda))
            print('Total number of running time is {overall_running_time:.3f}'.format(overall_running_time = overall_running_time))

            wandb.log({"train_loss":train_loss, 'val_loss':val_loss,
                    "train_prec1":train_prec1, "val_prec1":val_prec1,
                       "test_prec1": test_prec1 if test_loader is not None else val_prec1,
                "train_prec5":train_prec5, "val_prec5":val_prec5, "test_prec5": test_prec5 if test_loader is not None else val_prec5}, step = epoch)

            wandb.log({"lr": args.curlr, 'Optimized Lambda Variable': args.lamda}, step=epoch)
            wandb.log({"train loss": train_loss, 'train acc1': train_prec1, 'train acc5': train_prec5}, step=epoch)
            wandb.log({"s_t": mvg_g_obj.item(), 'moving_average_lambda': mvg_grad_lambda.item()}, step=epoch)
            wandb.log({"best test acc": best_acc1, 'beta': args.curbeta}, step=epoch)
            wandb.log({'training time':overall_running_time//60}, step=epoch)
            wandb.log({'scaling_factor': torch.exp(max_ivd_loss_prev - max_ivd_loss_cur) }, step=epoch)

            # if test_loader is not None:
            #     wandb.log({"test loss": test_loss, 'test acc1': test_prec1, 'test acc5': test_prec5}, step=epoch)
            # else:
            #     wandb.log({"test loss": val_loss, 'test acc1': val_prec1, 'test acc5': val_prec5}, step=epoch)
            #
def ACCSCCMA(args, model_new, model_prev, results):

    global w_grad_state
    w_grad_state = dict()
    train_loader, val_loader, test_loader = get_train_val_test_loader(args, None)
    if (args.frozen_aside_fc or args.frozen_aside_linear) and not args.not_frozen_last_block:
        train_loader, val_loader, test_loader = get_train_val_test_feature_loader(args)
        print("Datasets :", args.dataset, "Model Name :", args.model)

    ivd_criterion_cur = nn.CrossEntropyLoss(reduction='none')
    ivd_criterion_prev = nn.CrossEntropyLoss(reduction='none')
    CE_criterion = nn.CrossEntropyLoss()

    wandb.init(config = args, project="SCCMA", entity="qiqi-helloworld")
    train_loss, train_prec1, train_prec5= 0,0,0
    best_acc1 = 0
    start_time = time.time()
    args.lamda = args.lamda1
    mvg_g_obj, mv_lambda_u, mvg_grad_lambda, ivd_loss = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), 0

    test_prec1, test_prec5 = 0, 0
    if test_loader is not None:
        test_loss, test_prec1, test_prec5, test_auc_score = validate(args, test_loader, model_new, CE_criterion, 0)
        print('>>>>>>>> Pretrained test_prec1 {:.3f}'.format(test_prec1))


    for epoch in range(args.resumed_epoch, args.epochs):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        if args.epochs == 150 and epoch == 90:
            args.lrlambda =  args.lrlambda * 10

        elif args.epochs == 30 and epoch == 20:
            # args.lrlambda = args.lrlambda * 10
            pass
            # else: keep args.lrlamda constant


        model_new.train()
        model_prev.train()
        adjust_curlr_beta(epoch, args)
        lambda_cur, lambda_prev = args.lamda, args.lamda
        for batch_idx, (_, inputs, targets) in enumerate(train_loader):

            inputs, targets= inputs.cuda(), targets.cuda()

            outputs_cur, _ = model_new(inputs)
            ivd_loss_cur = ivd_criterion_cur(outputs_cur, targets)
            # max_ivd_loss_cur = torch.max(ivd_loss_cur).detach() /lambda_cur
            g_obj_cur = torch.exp(ivd_loss_cur/lambda_cur)
            mean_g_obj_cur = torch.mean(g_obj_cur)
            model_new.zero_grad()



            outputs_prev, _ = model_prev(inputs)
            ivd_loss_prev = ivd_criterion_prev(outputs_prev, targets)
            #max_ivd_loss_prev = torch.max(ivd_loss_prev).detach()/lambda_prev
            g_obj_prev = torch.exp(ivd_loss_prev / lambda_prev )
            mean_g_obj_prev = torch.mean(g_obj_prev)
            model_prev.zero_grad()

            # mean_g_obj_2.backward()
            (mean_g_obj_cur + mean_g_obj_prev).backward()
            # updates inner function objective g estimator
            mvg_g_obj = mean_g_obj_cur.item() + (1-args.curbeta)*(mvg_g_obj - mean_g_obj_prev.item())

            # updates v: inner function gradient estimator in terms of model parameter
            prev_grad_g_w = {}
            for k, v in model_prev.named_parameters():
                if v.requires_grad:
                    prev_grad_g_w.update({k: v.grad})

            # print(cur_grad_g_w.keys())
            for name, param in model_new.named_parameters():  # load the name and value of every layer.
              if name not in w_grad_state.keys() and param.requires_grad:
                  w_grad_state[name] = param.grad
                  # print(name, w_grad_state[name], w_grad_state[name].data)
              else:
                  if param.requires_grad:
                      # print(cur_grad_g_w[name].data)
                      w_grad_state[name] = param.grad +  (1-args.curbeta)* (w_grad_state[name] - prev_grad_g_w[name])
                      # updates u: inner function gradient estimator in terms of lambda
                      mv_lambda_u = -mean_g_obj_cur.item()*ivd_loss_cur.mean().item()/lambda_cur**2 + (1-args.curbeta)*(mv_lambda_u + mean_g_obj_prev.item()*ivd_loss_prev.mean().item()/lambda_prev**2)

            # copy the model parameters of current iteration to "model" for next iteration updates
            model_prev.load_state_dict(model_new.state_dict())
            #updates the new model
            for name, param in model_new.named_parameters():
                if param.requires_grad:
                    param.data.add_(-args.curlr,  (args.lamda/(mvg_g_obj))* w_grad_state[name] + args.weight_decay * param.data)

            # updates \lambda
            lambda_prev = args.lamda
            args.lamda = args.lamda - args.lrlambda * ((args.lamda/mvg_g_obj) * mv_lambda_u  + torch.log(mvg_g_obj) + args.rho + args.weight_decay * args.lamda)
            if args.lamda < args.lamda0:
                args.lamda = args.lamda0

            lambda_cur = args.lamda


            acc1, acc5 = accuracy(outputs_cur, targets, topk=(1, 5))
            loss = CE_criterion(outputs_cur, targets)
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))

            if batch_idx % args.print_freq == 0:

                if args.epochs <= 10:
                    train_loss, train_prec1, train_prc5, train_auc_score = validate(args, train_loader, model_new,
                                                                                    CE_criterion, epoch)
                    val_loss, val_prec1, val_prec5, val_auc_score = validate_cifar_val_loader(args, val_loader,
                                                                                              model_new,
                                                                                              CE_criterion, epoch)
                    print('iter acc1', epoch * len(train_loader) + batch_idx, 4 * len(train_loader), train_prec1,
                          val_prec1)
                    wandb.log({'iter acc1': train_prec1, 'iter val acc1': val_prec1},
                              step=epoch * len(train_loader) + batch_idx)
                else:
                    print('Epoch: [{0}][{1}/{2}]\t'
                              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                              'Train Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                              'Train Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                     epoch, batch_idx, len(train_loader),
                     loss=losses, top1=top1, top5=top5))


        if args.epochs > 10:
            train_loss, train_prec1, train_prc5, train_auc_score = validate(args, train_loader, model_new, CE_criterion, epoch)

            if 'cifar' in args.dataset:
                val_loss, val_prec1, val_prec5, val_auc_score = validate_cifar_val_loader(args, val_loader, model_new, CE_criterion, epoch)
            else:
                val_loss, val_prec1, val_prec5, val_auc_score = validate(args, val_loader, model_new, CE_criterion, epoch)

            if test_loader is not None:
                test_loss, test_prec1, test_prec5, test_auc_score = validate(args, test_loader, model_new, CE_criterion, epoch)

            overall_running_time = (time.time() - start_time)//60
            best_acc1 = max(best_acc1, val_prec1) if test_loader is None else max(best_acc1, test_prec1)
            results.add(epoch=epoch, val_loss=val_loss,
                        train_prec1 = train_prec1, val_prec1 = val_prec1,test_prec1 = test_prec1 if test_loader is not None else val_prec1,
                        train_prec5 = train_prec5, val_prec5 = val_prec5, test_prec5 = test_prec5 if test_loader is not None else val_prec5,
                        overall_running_time=overall_running_time)
            results.save()

            ##### Print on the screen.
            #if epoch % 15 == 0:
            output = ('Train: [{0}/{1}], lr: {lr:.5f}\t'
                          'Train Loss {train_loss:.4f} Val Loss {val_loss:.4f}\t'
                          'Train Prec@1 {train_prec1:.3f} Val Prec@1 {val_prec1:.3f} Test Prec@1 {test_prec1:.3f} \t'
                          'Train Prec@5 {train_prec5:.3f} Val Prec@5 {val_prec5:.3f} Test Prec@5 {test_prec5:.3f}'.format(
                    epoch, args.epochs, train_loss=train_loss, val_loss=val_loss,
                    train_prec1=train_prec1, val_prec1=val_prec1, test_prec1 = test_prec1 if test_loader is not None else val_prec1,
                train_prec5=train_prec5, val_prec5=val_prec5, test_prec5 = test_prec5 if test_loader is not None else val_prec5, lr = args.curlr))
            print(output)
            print("Lambda Variable value: ", str(args.lamda))
            print('Total number of running time is {overall_running_time:.3f}'.format(overall_running_time = overall_running_time))

            wandb.log({"lr": args.curlr, 'Optimized Lambda Variable': args.lamda}, step=epoch)
            wandb.log({"train loss": train_loss, 'train acc1': train_prec1, 'train acc5': train_prec5}, step=epoch)
            wandb.log({"s_t": mvg_g_obj.item(), 'moving_average_lambda': mvg_grad_lambda.item()}, step=epoch)
            wandb.log({"best test acc": best_acc1, 'beta': args.curbeta}, step=epoch)

            if test_loader is not None:
                wandb.log({"test loss": test_loss, 'test acc1': test_prec1, 'test acc5': test_prec5}, step=epoch)
            else:
                wandb.log({"test loss": val_loss, 'test acc1': val_prec1, 'test acc5': val_prec5}, step=epoch)



def myABSGD(args, model_new, results):
    global w_grad_state
    w_grad_state = dict()
    train_loader, val_loader, test_loader = get_train_val_test_loader(args, None)
    ivd_criterion_cur = nn.CrossEntropyLoss(reduction='none')
    CE_criterion = nn.CrossEntropyLoss()


    # initial lambda
    mylambda = args.mylambda
    if args.epochs == 200 or args.epochs == 300:
        sgd2absgd_epoch=160
    else:
        sgd2absgd_epoch = args.epochs //2
    abloss = ABLoss(mylambda, milestone = sgd2absgd_epoch, criterion=ivd_criterion_cur)
    optimizer = ABSGD(model_new.parameters(), lr = args.lr, weight_decay=args.weight_decay)
    # lrscheduler = MultiStepLR(optimizer=optimizer, milestones=[160, 180], gamma=0.01)
    wandb.init(config=args, project="SCCMA", entity="qiqi-helloworld")
    train_loss, train_prec1, train_prec5 = 0, 0, 0
    best_acc1 = 0
    start_time = time.time()
    print(optimizer.param_groups[0]['lr'])


    test_prec1, test_prec5 = 0, 0
    if test_loader is not None:
        test_loss, test_prec1, test_prec5, test_auc_score = validate(args, test_loader, model_new, CE_criterion, 0)
        print('>>>>>>>> Pretrained test_prec1 {:.3f}'.format(test_prec1))



    for epoch in range(args.resumed_epoch, args.epochs):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()


        model_new.train()
        adjust_curlr_beta(epoch, args)
        for batch_idx, (_, inputs, targets) in enumerate(train_loader):

            # print(inputs, targets)
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs, _ = model_new(inputs)
            ab_loss = abloss(outputs, targets)

            optimizer.zero_grad()
            ab_loss.backward()
            optimizer.step()



            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            loss = CE_criterion(outputs, targets)
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))

            if batch_idx % args.print_freq == 0:

                if args.epochs <= 10:

                    train_loss, train_prec1, train_prc5, train_auc_score = validate_cifar_val_loader(args, train_loader, model_new,
                                                                                    CE_criterion, epoch)
                    val_loss, val_prec1, val_prec5, val_auc_score = validate_cifar_val_loader(args, val_loader,
                                                                                              model_new,
                                                                                              CE_criterion, epoch)
                    print('iter acc1', epoch * len(train_loader) + batch_idx, 4 * len(train_loader), train_prec1,
                          val_prec1)
                    wandb.log({'iter acc1': train_prec1, 'iter val acc1': val_prec1},
                              step=epoch * len(train_loader) + batch_idx)
                else:
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Train Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Train Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        epoch, batch_idx, len(train_loader),
                        loss=losses, top1=top1, top5=top5))

        if args.epochs > 10:


            if 'cifar' in args.dataset:
                train_loss, train_prec1, train_prc5, train_auc_score = validate_cifar_val_loader(args, train_loader, model_new,
                                                                                CE_criterion,
                                                                                epoch)
                val_loss, val_prec1, val_prec5, val_auc_score = validate_cifar_val_loader(args, val_loader, model_new,
                                                                                          CE_criterion, epoch)
            else:
                train_loss, train_prec1, train_prc5, train_auc_score = validate(args, train_loader,
                                                                                                 model_new,
                                                                                                 CE_criterion,
                                                                                                 epoch)
                val_loss, val_prec1, val_prec5, val_auc_score = validate(args, val_loader, model_new, CE_criterion,
                                                                         epoch)

            if test_loader is not None:
                test_loss, test_prec1, test_prec5, test_auc_score = validate(args, test_loader, model_new, CE_criterion,
                                                                             epoch)

            overall_running_time = (time.time() - start_time) // 60
            best_acc1 = max(best_acc1, val_prec1) if test_loader is None else max(best_acc1, test_prec1)
            results.add(epoch=epoch, val_loss=val_loss,
                        train_prec1=train_prec1, val_prec1=val_prec1,
                        test_prec1=test_prec1 if test_loader is not None else val_prec1,
                        train_prec5=train_prec5, val_prec5=val_prec5,
                        test_prec5=test_prec5 if test_loader is not None else val_prec5,
                        overall_running_time=overall_running_time)
            results.save()

            ##### Print on the screen.
            # if epoch % 15 == 0:
            output = ('Train: [{0}/{1}], lr: {lr}\t'
                      'Train Loss {train_loss:.4f} Val Loss {val_loss:.4f}\t'
                      'Train Prec@1 {train_prec1:.3f} Val Prec@1 {val_prec1:.3f} Test Prec@1 {test_prec1:.3f} \t'
                      'Train Prec@5 {train_prec5:.3f} Val Prec@5 {val_prec5:.3f} Test Prec@5 {test_prec5:.3f}'.format(
                epoch, args.epochs, train_loss=train_loss, val_loss=val_loss,
                train_prec1=train_prec1, val_prec1=val_prec1,
                test_prec1=test_prec1 if test_loader is not None else val_prec1,
                train_prec5=train_prec5, val_prec5=val_prec5,
                test_prec5=test_prec5 if test_loader is not None else val_prec5, lr=optimizer.param_groups[0]['lr']))
            print(output)
            print("Lambda Variable value: ", str(args.mylambda))
            print('Total number of running time is {overall_running_time:.3f}'.format(
                overall_running_time=overall_running_time))

            wandb.log({"lr": optimizer.param_groups[0]['lr'], 'Optimized Lambda Variable': args.lamda}, step=epoch)
            wandb.log({"train loss": train_loss, 'train acc1': train_prec1, 'train acc5': train_prec5}, step=epoch)
            wandb.log({"best test acc": best_acc1, 'beta': args.curbeta}, step=epoch)

            if test_loader is not None:
                wandb.log({"test loss": test_loss, 'test acc1': test_prec1, 'test acc5': test_prec5}, step=epoch)
            else:
                wandb.log({"test loss": val_loss, 'test acc1': val_prec1, 'test acc5': val_prec5}, step=epoch)

        abloss.updateLambda()
        # lrscheduler.step()