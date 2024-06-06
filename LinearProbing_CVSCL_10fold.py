from __future__ import print_function

import os
import sys
import time
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.distributed as dist
import argparse
import socket
from torch.utils.data import distributed, Subset, SubsetRandomSampler, random_split
import tensorboard_logger as tb_logger
from torch.utils import data
from dataset_multi import RandomSampler, ConcatDataset

from torchvision import transforms, datasets
from dataset import RGB2Lab, RGB2YCbCr
from util import adjust_learning_rate, AverageMeter, accuracy

from models.alexnet import MyAlexNetCMC
from models.resnet import MyResNetsCMC
from models.LinearModel import LinearClassifierAlexNet, LinearClassifierResNet
from dataset import ImageFolderInstance
from sklearn.model_selection import KFold



def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=24, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=80, help='number of training epochs')

    # optimization
    # 注意: lr0.1,layer5 for alex; lr30,layer6 for r50v1/v2; lr50,layer6 for r50v3
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')  # 要改
    parser.add_argument('--lr_decay_epochs', type=str, default='10,20,30,40,50,60,70', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2, help='decay rate for learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')  # 要改
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')

    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    # model definition
    parser.add_argument('--model', type=str, default='alexnet', choices=['alexnet',
                                                                         'resnet50v1', 'resnet101v1', 'resnet18v1',
                                                                         'resnet50v2', 'resnet101v2', 'resnet18v2',
                                                                         'resnet50v3', 'resnet101v3', 'resnet18v3'])  # 要改
    parser.add_argument('--model_path', type=str,
                        default='/home/zhouhl/Documents/Code/CMC/model_path/casme3_mae/memory_softmax_2048_alexnet_lr_0.003_decay_0.0001_bsz_128_view_RGB_D/ckpt_epoch_110.pth',
                        # default='/home/zhouhl/Documents/Code/CMC/model_path/casme3_mae/memory_softmax_2048_resnet50v1_lr_0.003_decay_0.0001_bsz_128_view_RGB_D/ckpt_epoch_110.pth',
                        help='the model to test')  # 要改
    parser.add_argument('--layer', type=int, default=5, help='which layer to evaluate')  # 要改

    # dataset
    parser.add_argument('--dataset', type=str, default='casme3', choices=['imagenet100', 'imagenet', 'casme3'])

    # add new views
    parser.add_argument('--view', type=str, default='Lab', choices=['RGBD', 'Lab', 'YCbCr'])

    # path definition
    parser.add_argument('--data_folder1', type=str,
                        default='/home/zhouhl/Documents/data/casme3/cmc_test/casme3_diff_me4/casme3_diff_imgs_all',
                        help='path to data')
    parser.add_argument('--data_folder2', type=str,
                        default='/home/zhouhl/Documents/data/casme3/cmc_test/casme3_diff_me4/casme3_diff_depth_all',
                        help='path to data')
    parser.add_argument('--save_path', type=str,
                        default='/home/zhouhl/Documents/Code/CMC/model_path/casme3_linear_all',
                        help='path to save linear classifier')
    parser.add_argument('--tb_path', type=str,
                        default='/home/zhouhl/Documents/Code/CMC/tb_path/casme3_linear_all',
                        help='path to tensorboard')

    # data crop threshold
    parser.add_argument('--crop_low', type=float, default=0.2, help='low area in crop')

    # log file
    parser.add_argument('--log', type=str, default='time_linear.txt', help='log file')

    # GPU setting
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')

    opt = parser.parse_args()

    if (opt.data_folder1 is None) or (opt.data_folder2 is None) \
            or (opt.save_path is None) or (opt.tb_path is None):
        raise ValueError('one or more of the folders is None: data_folder1 | data_folder2 |save_path | tb_path')

    if opt.dataset == 'imagenet':
        if 'alexnet' not in opt.model:
            opt.crop_low = 0.08

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = opt.model_path.split('/')[-2]
    opt.model_name = 'calibrated_{}_bsz_{}_lr_{}_decay_{}'.format(opt.model_name, opt.batch_size, opt.learning_rate,
                                                                  opt.weight_decay)

    opt.model_name = '{}_view_{}'.format(opt.model_name, opt.view)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name + '_layer{}'.format(opt.layer))
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.save_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    if opt.dataset == 'imagenet100':
        opt.n_label = 100
    if opt.dataset == 'imagenet':
        opt.n_label = 1000
    if opt.dataset == 'casme3':
        opt.n_label = 4

    return opt


def get_train_val_loader(args):
    data_folder1 = os.path.join(args.data_folder1)
    data_folder2 = os.path.join(args.data_folder2)
    # val: '/home/zhouhl/Documents/data/casme3/casme3_diff_imgs', '/home/zhouhl/Documents/data/casme3/casme3_diff_depth

    train_transform1 = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(args.crop_low, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.4454, 0.4474, 0.4504], [0.4474, 0.4505, 0.4502]),
    ])
    train_transform2 = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(args.crop_low, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.0423, 0.0423, 0.0423], [0.1761, 0.176, 0.1761]),
    ])
    # val_transform1 = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.4454, 0.4474, 0.4504], [0.4474, 0.4505, 0.4502]),
    # ])
    # val_transform2 = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.0423, 0.0423, 0.0423], [0.1761, 0.176, 0.1761]),
    # ])

    dataset1 = ImageFolderInstance(data_folder1, transform=train_transform1)
    dataset2 = ImageFolderInstance(data_folder2, transform=train_transform2)
    new_dataset = ConcatDataset(dataset1, dataset2)

    # num_train = int(len(new_dataset)*0.9)
    # num_val = len(new_dataset)-num_train
    # train_dataset, val_dataset = random_split(new_dataset, [num_train, num_val])
    num_per = int(len(new_dataset)*0.1)
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = random_split(new_dataset, [num_per, num_per, num_per, num_per, num_per, num_per, num_per, num_per, num_per, len(new_dataset)-9*num_per])

    return x1, x2, x3, x4, x5, x6, x7, x8, x9, x10


def get_train_val_loader1(args, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, k):
    a = ConcatDataset(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10)

    val_dataset = a.datasets[k]
    if k == 0:
        train_dataset = ConcatDataset(a.datasets[1], a.datasets[2], a.datasets[3], a.datasets[4], a.datasets[5],
                               a.datasets[6], a.datasets[7], a.datasets[8], a.datasets[9])
    if k == 1:
        train_dataset = ConcatDataset(a.datasets[0], a.datasets[2], a.datasets[3], a.datasets[4], a.datasets[5],
                               a.datasets[6], a.datasets[7], a.datasets[8], a.datasets[9])
    if k == 2:
        train_dataset = ConcatDataset(a.datasets[0], a.datasets[1], a.datasets[3], a.datasets[4], a.datasets[5],
                               a.datasets[6], a.datasets[7], a.datasets[8], a.datasets[9])
    if k == 3:
        train_dataset = ConcatDataset(a.datasets[0], a.datasets[1], a.datasets[2], a.datasets[4], a.datasets[5],
                               a.datasets[6], a.datasets[7], a.datasets[8], a.datasets[9])
    if k == 4:
        train_dataset = ConcatDataset(a.datasets[0], a.datasets[1], a.datasets[2], a.datasets[3], a.datasets[5],
                               a.datasets[6], a.datasets[7], a.datasets[8], a.datasets[9])
    if k == 5:
        train_dataset = ConcatDataset(a.datasets[0], a.datasets[1], a.datasets[2], a.datasets[3], a.datasets[4],
                               a.datasets[6], a.datasets[7], a.datasets[8], a.datasets[9])
    if k == 6:
        train_dataset = ConcatDataset(a.datasets[0], a.datasets[1], a.datasets[2], a.datasets[3], a.datasets[4],
                               a.datasets[5], a.datasets[7], a.datasets[8], a.datasets[9])
    if k == 7:
        train_dataset = ConcatDataset(a.datasets[0], a.datasets[1], a.datasets[2], a.datasets[3], a.datasets[4],
                               a.datasets[5], a.datasets[6], a.datasets[8], a.datasets[9])
    if k == 8:
        train_dataset = ConcatDataset(a.datasets[0], a.datasets[1], a.datasets[2], a.datasets[3], a.datasets[4],
                               a.datasets[5], a.datasets[6], a.datasets[7], a.datasets[9])
    if k == 9:
        train_dataset = ConcatDataset(a.datasets[0], a.datasets[1], a.datasets[2], a.datasets[3], a.datasets[4],
                               a.datasets[5], a.datasets[6], a.datasets[7], a.datasets[8])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True)

    # # num of samples
    train_data = len(train_dataset)*9
    val_data = len(val_dataset)
    print('number of train_data: {}'.format(train_data))
    print('number of val_data: {}'.format(val_data))

    # return data_loader1, data_loader2, data_loader3, data_loader4, train_data, val_data
    return train_loader, val_loader, train_data, val_data


def set_model(args):
    if args.model.startswith('alexnet'):
        model = MyAlexNetCMC()
        classifier = LinearClassifierAlexNet(layer=args.layer, n_label=args.n_label, pool_type='max')
    elif args.model.startswith('resnet'):
        model = MyResNetsCMC(args.model)
        if args.model.endswith('v1'):
            classifier = LinearClassifierResNet(args.layer, args.n_label, 'avg', 1)
        elif args.model.endswith('v2'):
            classifier = LinearClassifierResNet(args.layer, args.n_label, 'avg', 2)
        elif args.model.endswith('v3'):
            classifier = LinearClassifierResNet(args.layer, args.n_label, 'avg', 4)
        else:
            raise NotImplementedError('model not supported {}'.format(args.model))
    else:
        raise NotImplementedError('model not supported {}'.format(args.model))

    # load pre-trained model
    print('==> loading pre-trained model')
    ckpt = torch.load(args.model_path)
    model.load_state_dict(ckpt['model'])
    print("==> loaded checkpoint '{}' (epoch {})".format(args.model_path, ckpt['epoch']))
    print('==> done')

    model = model.cuda()
    classifier = classifier.cuda()

    model.eval()

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    return model, classifier, criterion


def set_optimizer(args, classifier):
    optimizer = optim.SGD(classifier.parameters(),
                          lr=args.learning_rate,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(classifier.parameters(),
    #                              lr=args.learning_rate,
    #                              weight_decay=args.weight_decay,
    #                              betas=(args.beta1, args.beta2))
    return optimizer


def train(epoch, train_loader, model, classifier, criterion, optimizer, opt):
    """
    one epoch training
    """
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, dataloader in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs = dataloader[0][0]
        target = dataloader[0][1]

        inputs1 = dataloader[1][0]
        inputs1 = inputs1[:, 0:1, :, :]  # 取单通道

        new_inputs = torch.cat((inputs, inputs1), 1)

        input = new_inputs.float()
        if opt.gpu is not None:
            input = input.cuda(opt.gpu, non_blocking=True)
        target = target.cuda(opt.gpu, non_blocking=True)

        # ===================forward=====================
        with torch.no_grad():
            feat_rgb, feat_depth = model(input, opt.layer)
            feat = torch.cat((feat_rgb.detach(), feat_depth.detach()), dim=1)

        output = classifier(feat)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 3))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return top1.avg, top5.avg, losses.avg


def validate(val_loader, model, classifier, criterion, opt):
    """
    evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    classifier.eval()

    with torch.no_grad():
        end = time.time()
        for idx, dataloader in enumerate(val_loader):
            inputs = dataloader[0][0]
            target = dataloader[0][1]

            inputs1 = dataloader[1][0]
            inputs1 = inputs1[:, 0:1, :, :]  # 取单通道

            new_inputs = torch.cat((inputs, inputs1), 1)

            input = new_inputs.float()
            if opt.gpu is not None:
                input = input.cuda(opt.gpu, non_blocking=True)
            target = target.cuda(opt.gpu, non_blocking=True)

            # compute output
            feat_rgb, feat_depth = model(input, opt.layer)
            feat = torch.cat((feat_rgb.detach(), feat_depth.detach()), dim=1)
            output = classifier(feat)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 3))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                # print('Test: [{0}/{1}]\t'
                #       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                #       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                #       'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                #       'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                #        idx, len(val_loader), batch_time=batch_time, loss=losses,
                #        top1=top1, top5=top5))
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    idx, len(val_loader), batch_time=batch_time, loss=losses, top1=top1))

        print(' * Acc@1 {top1.avg:.3f} '
              .format(top1=top1))

    return top1.avg, top5.avg, losses.avg


def main():
    global best_acc1
    best_acc1 = 0

    args = parse_option()

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # set the data loader
    # train_loader, val_loader, train_data_num, val_data_num = get_train_val_loader(args)

    # set the model
    model, classifier, criterion = set_model(args)

    # set optimizer
    optimizer = set_optimizer(args, classifier)

    cudnn.benchmark = True

    # optionally resume linear classifier
    args.start_epoch = 1
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] + 1
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            classifier.load_state_dict(checkpoint['classifier'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    args.start_epoch = 1
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch'] + 1
            classifier.load_state_dict(checkpoint['classifier'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_acc1 = checkpoint['best_acc1']
            best_acc1 = best_acc1.cuda()
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
            torch.cuda.empty_cache()
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # tensorboard
    logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)

    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = get_train_val_loader(args)
    # routine
    for epoch in range(args.start_epoch, args.epochs + 1):

        adjust_learning_rate(epoch, args, optimizer)
        print("==> training...")

        time1 = time.time()
        train_acc = 0.0
        train_acc5 = 0.0
        train_loss = 0.0
        test_acc = 0.0
        test_acc5 = 0.0
        test_loss = 0.0
        for k in range(0, 10):
            train_loader, val_loader, train_data_num, val_data_num = get_train_val_loader1(args, x1, x2, x3, x4, x5, x6,
                                                                                           x7, x8, x9, x10, k)

            train_acc, train_acc5, train_loss = train(epoch, train_loader, model, classifier, criterion, optimizer, args)
            train_acc += train_acc
            train_acc5 += train_acc5
            train_loss += train_loss

            test_acc, test_acc5, test_loss = validate(val_loader, model, classifier, criterion, args)
            test_acc += test_acc
            test_acc5 += test_acc5
            test_loss += test_loss

        train_acc /= 10
        train_acc5 /= 10
        train_loss /= 10
        test_acc /= 10
        test_acc5 /= 10
        test_loss /= 10
        time2 = time.time()
        print('train epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_acc5', train_acc5, epoch)
        logger.log_value('train_loss', train_loss, epoch)

        print("==> testing...")
        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_acc5', test_acc5, epoch)
        logger.log_value('test_loss', test_loss, epoch)

        # save the best model
        if test_acc > best_acc1:
            best_acc1 = test_acc
            state = {
                'opt': args,
                'epoch': epoch,
                'classifier': classifier.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }
            save_name = '{}_layer{}.pth'.format(args.model, args.layer)
            save_name = os.path.join(args.save_folder, save_name)
            print('saving best model!')
            torch.save(state, save_name)

        # save model
        if epoch % args.save_freq == 0:
            print('==> Saving...')
            state = {
                'opt': args,
                'epoch': epoch,
                'classifier': classifier.state_dict(),
                'best_acc1': test_acc,
                'optimizer': optimizer.state_dict(),
            }
            save_name = 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch)
            save_name = os.path.join(args.save_folder, save_name)
            print('saving regular model!')
            torch.save(state, save_name)

        # tensorboard logger
        pass


if __name__ == '__main__':
    best_acc1 = 0
    main()
